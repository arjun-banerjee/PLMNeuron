import json
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc

# Config
ESM_NAME = "facebook/esm2_t12_35M_UR50D"
#ESM_NAME = "facebook/esm2_t6_8M_UR50D"
BATCH_SIZE = 2048  # Start smaller
K = 30
PARQUET_PATH = "../datatest573230.parquet"

device = "cuda" if torch.cuda.is_available() else "cpu"

def process_sequences_single_pass():
    """Process all sequences in a single forward pass per batch"""
    
    # Load model once
    tokenizer = EsmTokenizer.from_pretrained(ESM_NAME)
    model = EsmForMaskedLM.from_pretrained(ESM_NAME).to(device)
    model = model.half()
    model.eval()
    
    n_layers = len(model.esm.encoder.layer)
    H = model.config.hidden_size
    
    # Store all activations first, then find top-k
    all_activations = []  # Will store: [(layer_idx, batch_start_idx, activations)]
    
    # Hook to capture all layer outputs
    layer_outputs = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Store CLS token activations
            layer_outputs[layer_idx] = output[:, 0, :].detach().cpu().numpy()
        return hook
    
    # Register hooks for all layers
    handles = []
    for i, layer in enumerate(model.esm.encoder.layer):
        handle = layer.register_forward_hook(make_hook(i))
        handles.append(handle)

    # load and process data from hugging face
    dataset = load_dataset("camillexdang/plminterp")
    df = dataset["train"].to_pandas()
    df = df.drop_duplicates(subset=['Sequence']).reset_index(drop=True)
    
    # Load and process data
    # df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    # df = df.drop_duplicates(subset=['Sequence']).reset_index(drop=True)
    dataset = df["Sequence"].tolist()
    
    print(f"Processing {len(dataset)} sequences")
    
    # Process in batches - single forward pass captures all layers
    for start in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch_seq = dataset[start:start+BATCH_SIZE]
        inputs = tokenizer(batch_seq, return_tensors="pt", 
                          padding=True, truncation=True, max_length=512)  # Reduced max length
        
        with torch.no_grad():
            _ = model(input_ids=inputs["input_ids"].to(device), 
                     attention_mask=inputs["attention_mask"].to(device))
        
        # Store activations from all layers
        for layer_idx in range(n_layers):
            if layer_idx in layer_outputs:
                all_activations.append((layer_idx, start, layer_outputs[layer_idx].copy()))
        
        layer_outputs.clear()
        torch.cuda.empty_cache()
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Computing top-k activations...")
    
    # Find top-k for each layer and hidden dimension
    results = {}
    for layer_idx in tqdm(range(n_layers), desc="Processing layers"):
        results[layer_idx] = {"top_k_high": {}, "top_k_low": {}}
        
        # Collect all activations for this layer
        layer_activations = []
        layer_indices = []
        
        for l_idx, batch_start, activations in all_activations:
            if l_idx == layer_idx:
                layer_activations.append(activations)
                batch_indices = np.arange(batch_start, batch_start + activations.shape[0])
                layer_indices.append(batch_indices)
        
        if layer_activations:
            # Concatenate all batches for this layer
            full_activations = np.concatenate(layer_activations, axis=0)  # (n_sequences, hidden_size)
            full_indices = np.concatenate(layer_indices)
            
            for h in range(H):
                scores = full_activations[:, h]
                
                # Top-k highest
                top_high_idx = np.argpartition(scores, -K)[-K:]
                top_high_idx = top_high_idx[np.argsort(scores[top_high_idx])[::-1]]
                
                results[layer_idx]["top_k_high"][h] = [
                    {"sequence": dataset[full_indices[idx]], "activation": float(scores[idx])}
                    for idx in top_high_idx
                ]
                
                # Top-k lowest
                top_low_idx = np.argpartition(scores, K)[:K]
                top_low_idx = top_low_idx[np.argsort(scores[top_low_idx])]
                
                results[layer_idx]["top_k_low"][h] = [
                    {"sequence": dataset[full_indices[idx]], "activation": float(scores[idx])}
                    for idx in top_low_idx
                ]
    
    return results

if __name__ == "__main__":
    results = process_sequences_single_pass()
    
    # Save results
    with open("esm35M_500kdataset_k30_optimized.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Exemplar search complete!")