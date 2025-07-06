#!/usr/bin/env python3
"""
simulator_pipeline.py (refactored for COVID-19 tweet classification)

Fine-tunes a sequence-level classifier to predict if a tweet is coronavirus-related (binary classification)
using the Kaggle COVID-19 NLP dataset. Uses the same model, parameters, and Trainer structure as before.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import kagglehub

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
OUTPUT_DIR       = "simulator_finetune"
STEPS            = 100      # total number of training epochs
BATCH_SIZE       = 16
SPLIT_RATIO      = 0.8
LOG_STEPS        = 2        # how often to log/evaluate/save
MODEL_NAME = "allenai/longformer-base-4096"

# Download and load Kaggle dataset
path = kagglehub.dataset_download("datatattle/covid-19-nlp-text-classification")
print("Path to dataset files:", path)
train_csv = os.path.join(path, "Corona_NLP_train.csv")
df = pd.read_csv(train_csv, encoding="latin1")

# Preprocess: 5-class label (0=extremely negative, 1=negative, 2=neutral, 3=positive, 4=extremely positive)
def label_fn(sentiment):
    s = sentiment.strip().lower()
    if s == "extremely negative":
        return 0
    elif s == "negative":
        return 1
    elif s == "neutral":
        return 2
    elif s == "positive":
        return 3
    elif s == "extremely positive":
        return 4
    else:
        return 2  # fallback to neutral

df = df.dropna(subset=["OriginalTweet", "Sentiment"])
df["label"] = df["Sentiment"].apply(label_fn)

# Tokenizer
MAX_LEN = 4096
# (Longformer can handle long text, but tweets are short)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Build examples
def build_examples(df):
    examples = []
    for _, row in df.iterrows():
        text = row["OriginalTweet"]
        label = row["label"]
        prompt = (
            "You are a sentiment analysis model. "
            "Classify the following tweet about coronavirus on a scale of 0-4, where: "
            "0 = extremely negative, 1 = negative, 2 = neutral, 3 = positive, 4 = extremely positive.\n\n"
            f"TWEET: {text}\n\nPrediction (0-4):"
        )
        examples.append({"text": prompt, "label": label})
    return examples

all_exs = build_examples(df)
random.shuffle(all_exs)
split = int(SPLIT_RATIO * len(all_exs))
steps_per_epoch = max(split // BATCH_SIZE, 1)
train_exs, val_exs = all_exs[:split], all_exs[split:]

# Dataset class
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

# Metrics
def metrics(pred):
    try:
        preds = torch.from_numpy(pred.predictions).argmax(dim=-1) if isinstance(pred.predictions, np.ndarray) else pred.predictions.argmax(dim=-1)
        labs = pred.label_ids
        preds = preds.flatten().cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds).flatten()
        labs  = labs.flatten().cpu().numpy()  if isinstance(labs,  torch.Tensor) else np.array(labs).flatten()
        acc = (preds == labs).mean()
        print(f"[METRICS] Accuracy = {acc:.4f}")
        return {"accuracy": acc}
    except Exception as e:
        print(f"[METRICS ERROR] Failed to compute accuracy: {e}")
        return {"accuracy": 0.0}

# Training

def run_finetune(out_dir, train_exs, val_exs):
    print("Starting fine-tuning...")
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5
    ).to(device)
    train_ds = make_dataset(train_exs, tokenizer)
    val_ds   = make_dataset(val_exs,   tokenizer)
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
        dataloader_num_workers=2,
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
    # Extract metrics
    history = trainer.state.log_history
    train_steps = [h["step"] for h in history if "loss" in h]
    train_loss  = [h["loss"] for h in history  if "loss" in h]
    eval_steps  = [h["step"] for h in history if "eval_loss" in h]
    eval_loss   = [h["eval_loss"] for h in history if "eval_loss" in h]
    acc_steps = [h["step"]           for h in history if "eval_accuracy" in h]
    acc_vals  = [h["eval_accuracy"] for h in history if "eval_accuracy" in h]
    # Plot train vs eval loss
    plt.figure()
    plt.plot(train_steps, train_loss, label="Train Loss")
    plt.plot(eval_steps,  eval_loss,  label="Eval Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Train vs Eval Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_comparison.png"))
    # Plot Accuracy
    plt.figure()
    plt.plot(acc_steps, acc_vals)
    plt.xlabel("Step")
    plt.ylabel("Eval Accuracy")
    plt.title("Simulator Fine-tune Accuracy")
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))

if __name__ == '__main__':
    run_finetune(OUTPUT_DIR, train_exs, val_exs)

# --- UNUSED GLOBALS FOR LEGACY FUNCTIONS (DUMMY VALUES) ---
import json  # for load_activations
ACTIVATIONS_JSON = "unused_activations.json"
neuron_maxes = {}
EVAL_TOP_N = 6
EVAL_BOTTOM_N = 2
NEURON_SUBSAMPLE_RATIO = 50

# --- UNUSED FUNCTIONS FROM ORIGINAL SCRIPT (PRESERVED) ---

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

from transformers import TrainerCallback
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

def build_examples_old(acts, expl, dataset):
    print("Building examples with multiprocessing...")
    ds_unique = dataset.drop_duplicates(subset=['Sequence'], keep='first')
    dataset_dict = ds_unique.set_index('Sequence').to_dict('index')
    rows = [row for i, (_, row) in enumerate(expl.iterrows()) if i % NEURON_SUBSAMPLE_RATIO == 0]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
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

