import json
import heapq
import torch
from nnsight import LanguageModel
from transformers import EsmTokenizer, EsmForMaskedLM
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import multiprocessing as mp
import numpy as np

# Configurations
ESM_NAME = "facebook/esm2_t6_8M_UR50D"
BATCH_SIZE = 4
K = 100
NUM_PROCESSES = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_parquet("data1mil.parquet", engine="pyarrow")
dataset = df["Sequence"].tolist()[:1500]

tokenizer = EsmTokenizer.from_pretrained(ESM_NAME)
esm_model = EsmForMaskedLM.from_pretrained(ESM_NAME).to(device)
lm = LanguageModel(esm_model, tokenizer=tokenizer)

n_layers = len(esm_model.esm.encoder.layer)
H = esm_model.config.hidden_size

def merge_heaps(global_heaps, local_heaps):
    for l in range(n_layers):
        for h in range(H):
            for score, idx in local_heaps[l]["high"][h]:
                heapq.heappush(global_heaps[l]["high"][h], (score, idx))
                if len(global_heaps[l]["high"][h]) > K:
                    heapq.heappop(global_heaps[l]["high"][h])
            for score, idx in local_heaps[l]["low"][h]:
                heapq.heappush(global_heaps[l]["low"][h], (score, idx))
                if len(global_heaps[l]["low"][h]) > K:
                    heapq.heappop(global_heaps[l]["low"][h])
    return global_heaps

def process_shard(proc_id, shard, return_dict):
    local_heaps = [
        {"high": [[] for _ in range(H)], "low": [[] for _ in range(H)]}
        for _ in range(n_layers)
    ]

    for start in tqdm(range(0, len(shard), BATCH_SIZE), desc=f"Process {proc_id}", position=proc_id):
        batch_seq = shard[start:start+BATCH_SIZE]
        inputs = tokenizer(batch_seq, return_tensors="pt", padding=True, truncation=True)
        batch_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            with lm.trace(batch_ids) as tracer:
                _ = lm(input_ids=batch_ids, attention_mask=attention_mask)
                saved = [lm.esm.encoder.layer[i].nns_output.save() for i in range(n_layers)]
        
        activations = [p.value for p in saved]

        for layer_idx, tensor in enumerate(activations):
            tensor = tensor[0]
            scores = tensor.max(dim=1).values.cpu()
            scores_np = scores.numpy()
            global_indices = np.arange(start, start + scores.shape[0]) + proc_id * len(shard)

            for h in range(H):
                neuron_scores = scores_np[:, h]
                top_k_high = heapq.nlargest(K, zip(neuron_scores, global_indices))
                top_k_low = heapq.nlargest(K, zip(-neuron_scores, global_indices))

                high_heap = local_heaps[layer_idx]["high"][h]
                low_heap = local_heaps[layer_idx]["low"][h]

                for score, idx in top_k_high:
                    if len(high_heap) < K:
                        heapq.heappush(high_heap, (score, idx))
                    elif score > high_heap[0][0]:
                        heapq.heappushpop(high_heap, (score, idx))

                for neg_score, idx in top_k_low:
                    if len(low_heap) < K:
                        heapq.heappush(low_heap, (neg_score, idx))
                    elif neg_score > low_heap[0][0]:
                        heapq.heappushpop(low_heap, (neg_score, idx))

    return_dict[proc_id] = local_heaps

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    shard_size = len(dataset) // NUM_PROCESSES
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []

    for i in range(NUM_PROCESSES):
        shard = dataset[i * shard_size : (i + 1) * shard_size]
        p = mp.Process(target=process_shard, args=(i, shard, return_dict))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    final_heaps = [
        {"high": [[] for _ in range(H)], "low": [[] for _ in range(H)]}
        for _ in range(n_layers)
    ]
    for local_heaps in return_dict.values():
        merge_heaps(final_heaps, local_heaps)

    results = {}
    for l in range(n_layers):
        results[l] = {"top_k_high": {}, "top_k_low": {}}
        for h in range(H):
            high_list = heapq.nlargest(K, final_heaps[l]["high"][h])
            low_list = heapq.nlargest(K, final_heaps[l]["low"][h])
            results[l]["top_k_high"][h] = [
                {"sequence": dataset[idx], "activation": float(score)}
                for score, idx in high_list
            ]
            results[l]["top_k_low"][h] = [
                {"sequence": dataset[idx], "activation": float(-score)}
                for score, idx in low_list
            ]

    with open("esm2_neuron_exemplars_parallel.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Parallel exemplar search complete.")
