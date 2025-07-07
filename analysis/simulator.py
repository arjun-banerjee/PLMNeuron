#!/usr/bin/env python3
"""
simulator_pipeline.py

Single-stage fine-tuning of a sequence-level simulator that predicts normalized activation
scores (0–10) from (neuron hypothesis, sequence, features), with midpoint and final checkpoints,
periodic evaluation, and combined metric plots.
"""

import os
import json
import random
import numpy as np


import pandas as pd
import torch
from scipy.stats import pearsonr
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from transformers import TrainerCallback



# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
ACTIVATIONS_JSON = "../esm8M_500kdataset_k100_optimized.json"
EXPLANATIONS_CSV = "esm8M_500k_neuron_explanations.csv"
DATASET_PARQUET  = "../datatest573230.parquet"
OUTPUT_DIR       = "simulator_finetune"
EVAL_TOP_N       = 6
EVAL_BOTTOM_N    = 2
STEPS            = 100      # total number of training epochs
BATCH_SIZE       = 16
SPLIT_RATIO      = 0.8
LOG_STEPS        = 2        # how often to log/evaluate/save
NEURON_SUBSAMPLE_RATIO = 50  # Keep only 1 in every 20 neurons


# Base model
MODEL_NAME = "allenai/longformer-base-4096"
# MODEL_NAME = "google/bigbird-roberta-base" # try this too

# Initialize tokenizer to get model_max_length
tokenizer_tmp = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 4096
# Cache for neuron max activations
neuron_maxes = {}

# Keys to keep for feature compression
KEYS_TO_KEEP = [
    "length", "Mass", "mol_weight", "iso_point", "gravy", "charge_pH7",
    "helix_frac", "turn_frac", "sheet_frac", "instability_index",
    "boman_index", "aliphatic_index", "hydrophobic_moment",
    "Protein names", "Organism", "Subcellular location [CC]",
    "Gene Ontology (biological process)", "Gene Ontology (molecular function)",
    "Function [CC]", "Disruption phenotype", "Catalytic activity", "Pathway"
]

def compress_features(feat):
    comp = {}
    for k in KEYS_TO_KEEP:
        v = feat.get(k)
        if isinstance(v, float):
            v = round(v, 4)
        if v is None or (isinstance(v, str) and v.lower() == "nan"):
            continue
        comp[k] = v
    return comp

def load_activations():
    print("Loading activations...")
    with open(ACTIVATIONS_JSON) as f:
        acts = json.load(f)
    for layer_str, data in tqdm(acts.items(), desc="Layers"):
        layer = int(layer_str)
        neuron_maxes[layer] = {}
        highs = data.get("top_k_high", {})
        lows  = data.get("top_k_low", {})
        for n_str, entries in highs.items():
            neuron = int(n_str)
            vals = [e["activation"] for e in entries]
            vals += [e["activation"] for e in lows.get(n_str, [])]
            neuron_maxes[layer][neuron] = max(vals) if vals else 1.0
    print("Done loading activations.")
    return acts

class ManualEvalCallback(TrainerCallback):
    def __init__(self, trainer_ref, val_dataset):
        self.trainer_ref = trainer_ref
        self.val_dataset = val_dataset
        self.eval_results = []

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n[ManualEvalCallback] Epoch {state.epoch:.2f} finished. Running manual eval...")
        metrics = self.trainer_ref.evaluate(eval_dataset=self.val_dataset)
        self.eval_results.append((state.global_step, metrics))
        return control

def process_chunk_worker(chunk_data):
    chunk_expl, acts, dataset_dict = chunk_data
    local_exs = []
    for row in chunk_expl:
        layer, neuron = map(int, row['neuron_id'].split('_')[1::2])
        hypo = row['explanation_1']
        layer_acts = acts.get(str(layer), {})
        top = layer_acts.get("top_k_high", {}).get(str(neuron), [])[:EVAL_TOP_N]
        bot = layer_acts.get("top_k_low",  {}).get(str(neuron), [])[:EVAL_BOTTOM_N]
        max_act = neuron_maxes[layer].get(neuron, 1.0)
        for item in top + bot:
            seq = item['sequence']
            raw = item['activation']
            norm = raw / max_act if max_act else 0.0
            norm = min(max(norm, 0.0), 1.0)
            label = int(round(norm * 10))
            feat = dataset_dict.get(seq, {})
            comp = compress_features(feat)
            text = (
                f"You are a protein language model neuron simulator. Your task is to predict how strongly "
                f"a specific neuron will activate when given a protein sequence.\n\n"
                f"NEURON INFORMATION:\n"
                f"- Neuron ID: {row['neuron_id']}\n"
                f"- Hypothesis: This neuron is hypothesized to detect: {hypo}\n\n"
                f"PROTEIN SEQUENCE:\n{seq}\n\n"
                f"PROTEIN FEATURES:\n{comp}\n\n"
                f"TASK: Based on the neuron's hypothesized function and the protein's sequence/features, "
                f"predict how strongly this neuron will activate on a scale of 0-10, where:\n"
                f"- 0 = No activation (neuron doesn't respond to this protein)\n"
                f"- 5 = Moderate activation\n"
                f"- 10 = Maximum activation (protein strongly matches neuron's hypothesized function)\n\n"
                f"Prediction (0-10):"
            )
            local_exs.append({'text': text, 'label': label})
    return local_exs

def build_examples(acts, expl, dataset):
    print("Building examples with multiprocessing...")
    ds_unique = dataset.drop_duplicates(subset=['Sequence'], keep='first')
    dataset_dict = ds_unique.set_index('Sequence').to_dict('index')
    rows = [row for i, (_, row) in enumerate(expl.iterrows()) if i % NEURON_SUBSAMPLE_RATIO == 0]
    n_procs = min(mp.cpu_count(), len(rows))
    chunk_size = (len(rows) + n_procs - 1) // n_procs
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    chunk_data = [(chunk, acts, dataset_dict) for chunk in chunks]
    exs = []
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for part in tqdm(executor.map(process_chunk_worker, chunk_data), total=len(chunk_data), desc="Chunks"):
            exs.extend(part)
    print(f"Built {len(exs)} examples.")
    return exs

def make_dataset(exs, tokenizer):
    class SeqDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.examples = exs
            self.tokenizer = tokenizer
            self.max_len = MAX_LEN
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            t = self.examples[idx]
            enc = self.tokenizer(
                t['text'],
                truncation=True, padding='max_length',
                max_length=self.max_len, return_tensors='pt'
            )
            item = {k: v.squeeze() for k, v in enc.items()}
            item['labels'] = torch.tensor(t['label'], dtype=torch.long)
            return item
    return SeqDataset()

def metrics(pred):
    try:
        preds = torch.from_numpy(pred.predictions).argmax(dim=-1) if isinstance(pred.predictions, np.ndarray) else pred.predictions.argmax(dim=-1)
        labs = pred.label_ids

        # Convert everything to flat NumPy arrays
        preds = preds.flatten().cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds).flatten()
        labs  = labs.flatten().cpu().numpy()  if isinstance(labs,  torch.Tensor) else np.array(labs).flatten()

        print(f"pred.predictions.shape: {pred.predictions.shape}")
        print(f"pred.predictions[:5]: {pred.predictions[:5]}")

        print(f"[METRICS] preds[:5]: {preds[:5]}")
        print(f"[METRICS] labs[:5]:  {labs[:5]}")

        if len(labs) > 1 and len(preds) == len(labs):
            r, _ = pearsonr(labs, preds)
        else:
            r = 0.0
        print(f"[METRICS] Pearson r = {r:.4f}")
        return {"pearson_r": r}

    except Exception as e:
        print(f"[METRICS ERROR] Failed to compute Pearson: {e}")
        return {"pearson_r": 0.0}


def run_finetune(out_dir, train_exs, val_exs):
    print("Starting fine-tuning...")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=11
    ).to(device)

    train_ds = make_dataset(train_exs, tokenizer)
    val_ds   = make_dataset(val_exs,   tokenizer)

    # midpoint in steps = (total_epochs * steps_per_epoch) // 2
    # but here we simply save at half the epochs
    midpoint = STEPS // 2

    args = TrainingArguments(
        output_dir=out_dir,
        do_train=True,
        do_eval=True,
        logging_steps=steps_per_epoch,
        eval_steps=   steps_per_epoch,
        save_steps=   steps_per_epoch,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=STEPS,
        fp16=True,
        dataloader_num_workers=mp.cpu_count(),
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=metrics
    )

    eval_callback = ManualEvalCallback(trainer, val_ds)
    trainer.add_callback(eval_callback)

    trainer.train()
    print("Fine-tuning complete.")

    # Extract metrics
    history = trainer.state.log_history

    train_steps = [h["step"] for h in history if "loss" in h]
    train_loss  = [h["loss"] for h in history  if "loss" in h]

    eval_steps  = [h["step"] for h in history if "eval_loss" in h]
    eval_loss   = [h["eval_loss"] for h in history if "eval_loss" in h]

    pearson_steps = [h["step"]           for h in history if "eval_pearson_r" in h]
    pearson_vals  = [h["eval_pearson_r"] for h in history if "eval_pearson_r" in h]

    # Plot train vs eval loss
    plt.figure()
    plt.plot(train_steps, train_loss, label="Train Loss")
    plt.plot(eval_steps,  eval_loss,  label="Eval Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Train vs Eval Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_comparison.png"))

    # Plot Pearson-r
    plt.figure()
    plt.plot(pearson_steps, pearson_vals)
    plt.xlabel("Step")
    plt.ylabel("Eval Pearson r")
    plt.title("Simulator Fine-tune Pearson")
    plt.savefig(os.path.join(out_dir, "pearson_curve.png"))

if __name__ == '__main__':
    acts = load_activations()
    expl = pd.read_csv(EXPLANATIONS_CSV)
    ds   = pd.read_parquet(DATASET_PARQUET, engine='pyarrow').drop_duplicates(subset=['Sequence'], keep='first')

    all_exs = build_examples(acts, expl, ds)
    random.shuffle(all_exs)
    split = int(SPLIT_RATIO * len(all_exs))
    steps_per_epoch = max(split // BATCH_SIZE, 1)

    train_exs, val_exs = all_exs[:split], all_exs[split:]

    run_finetune(OUTPUT_DIR, train_exs, val_exs)

