#!/usr/bin/env python3
"""
simulator_pipeline.py

Single-stage fine-tuning of a sequence-level simulator that predicts normalized activation
scores (0–10) from (neuron hypothesis, sequence, features), with only two checkpoints
(midpoint and final) and GPU optimizations.
"""

import os
import json
import random
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
K_HYPOTHESES     = 2
STEPS            = 25       # total number of training epochs
BATCH_SIZE       = 16
SPLIT_RATIO      = 0.8
LOG_STEPS        = 5      # how often to log/evaluate/save

# Initialize tokenizer to get model_max_length
MODEL_NAME = "google/electra-base-discriminator"
tokenizer_tmp = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = tokenizer_tmp.model_max_length

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
                "Task: Predict activation 0–10. ONLY ANSWER WITH A NUMBER\n"
                f"Neuron: {row['neuron_id']}\n"
                f"Description: {hypo}\n"
                f"Sequence: {seq}\n"
                f"Features: {comp}\n"
                "ONLY ANSWER WITH A NUMBER BETWEEN 0 AND 10."
            )
            local_exs.append({'text': text, 'label': label})
    return local_exs

def build_examples(acts, expl, dataset):
    print("Building examples with multiprocessing...")
    ds_unique = dataset.drop_duplicates(subset=['Sequence'], keep='first')
    dataset_dict = ds_unique.set_index('Sequence').to_dict('index')
    rows = [row for _, row in expl.iterrows()]
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
            item = {k: v.squeeze().to(device) for k, v in enc.items()}
            item['labels'] = torch.tensor(t['label'], dtype=torch.long, device=device)
            return item
    return SeqDataset()

def metrics(pred):
    preds = pred.predictions.argmax(-1)
    labs  = pred.label_ids
    r, _ = pearsonr(labs, preds) if len(labs) > 1 else (0.0, None)
    return {'pearson_r': r}

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

    # Calculate half-epoch checkpoint step
    save_step = STEPS // 2

    args = TrainingArguments(
        output_dir=out_dir,
        do_train=True,
        do_eval=True,
        eval_steps=LOG_STEPS,
        logging_steps=LOG_STEPS,
        save_steps=LOG_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=STEPS,
        fp16=True,                  # mixed precision
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

    trainer.train()
    print("Fine-tuning complete.")

    # Plot Pearson-r over time
    history = trainer.state.log_history
    steps = [h['step'] for h in history if 'eval_pearson_r' in h]
    pearson = [h['eval_pearson_r'] for h in history if 'eval_pearson_r' in h]
    plt.figure()
    plt.plot(steps, pearson)
    plt.xlabel('Training Step')
    plt.ylabel('Eval Pearson r')
    plt.title('Simulator Fine-tune')
    plt.savefig(os.path.join(out_dir, "pearson_curve.png"))

if __name__ == '__main__':
    print("loading acts")
    acts = load_activations()
    print("acts loaded")
    expl = pd.read_csv(EXPLANATIONS_CSV)
    print("expl loaded")
    ds   = pd.read_parquet(DATASET_PARQUET, engine='pyarrow').drop_duplicates(subset=['Sequence'], keep='first')
    print("ds loaded")

    all_exs = build_examples(acts, expl, ds)
    random.shuffle(all_exs)
    split = int(SPLIT_RATIO * len(all_exs))
    train_exs, val_exs = all_exs[:split], all_exs[split:]

    run_finetune(OUTPUT_DIR, train_exs, val_exs)
    print("Done: checkpoints are in", OUTPUT_DIR)
    trainer.save_model("savedmodel")
    print("Final model saved to", OUTPUT_DIR)

