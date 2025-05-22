import json
import heapq
import torch
from nnsight import LanguageModel
from transformers import EsmTokenizer, EsmForMaskedLM
from tqdm import tqdm
import pandas as pd

print("All improts done")
device = "cuda" if torch.cuda.is_available() else "cpu"

#Configurations
ESM_NAME = "facebook/esm2_t6_8M_UR50D"
BATCH_SIZE = 4
K = 10


#Dataset
df = pd.read_csv("data_collector.csv")
dataset = df["Sequence"].tolist()
dataset = dataset[:512]

print("All data loaded")

#1 -- Load tokenizer and base model without LM head
tokenizer = EsmTokenizer.from_pretrained(ESM_NAME)
esm_model = EsmForMaskedLM.from_pretrained(ESM_NAME)

print("Tokenzier and model loaded")

#2 -- Wrap with NNSight for tracing
lm = LanguageModel(esm_model, tokenizer=tokenizer)
print("NNSight wrapped")

#3 -- Inspect archetecture
layers = esm_model.esm.encoder.layer
n_layers = len(layers)
H = esm_model.config.hidden_size


#4 -- Heaps -- 1 per layer x neuron
heaps = [
    {"high": [[] for _ in range(H)],
     "low": [[] for _ in range(H)]}
    for _ in range(n_layers)
]

print("Heaps made")
#5 -- stream through mini batches

print("Starting inference")
for start in tqdm(range(0, len(dataset), BATCH_SIZE)):
    batch_seq = dataset[start:start+BATCH_SIZE] 
    inputs = tokenizer(batch_seq, return_tensors="pt", padding=True, truncation=True)
    batch_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    #Trace this minibatch
    with lm.trace(batch_ids) as tracer:
        output = lm(input_ids=batch_ids, attention_mask=attention_mask)
        saved = [lm.esm.encoder.layer[i].nns_output.save() for i in range(n_layers  )]
        
    activations = [p.value for p in saved]  # each p.value is the [B, L, H] tensor
    for layer_idx, tensor in enumerate(activations):
        #[B, L, H] -> max over tokens -> [B, H]
        tensor = tensor[0]
        scores = tensor.max(dim=1).values
        for local_idx in range(scores.size(0)):
            global_idx = start + local_idx
            for h in range(H):
                val = float(scores[local_idx, h])
                #top K highest
                hh = heaps[layer_idx]["high"][h]
                if len(hh) < K:
                    heapq.heappush(hh, (val, global_idx))
                elif val > hh[0][0]:
                    heapq.heappushpop(hh, (val, global_idx))
                #top-K lowest via negation
                ll = heaps[layer_idx]["low"][h]
                neg = -val
                if len(ll) < K:
                    heapq.heappush(ll, (neg, global_idx))
                elif neg > ll[0][0]:
                    heapq.heappushpop(ll, (neg, global_idx))

print("Inference done")
print("Finding exemplars")
# 6=--  After streaming all batches, extract and save exemplars
results = {}
for l in range(n_layers):
    results[l] = {"top_k_high": {}, "top_k_low": {}}
    for h in range(H):
        high_list = heapq.nlargest(K, heaps[l]["high"][h])
        low_list  = heapq.nlargest(K, heaps[l]["low"][h])
        
         
        results[l]["top_k_high"][h] = [
            {"sequence": dataset[idx], "activation": score}
            for score, idx in high_list
        ]
        results[l]["top_k_low"][h] = [
            {"sequence": dataset[idx], "activation": -score}
            for score, idx in low_list
        ]

print("Saving file")

with open("esm2_neuron_exemplars.json", "w") as f:
    json.dump(results, f, indent=2)
      
                    
                


        
        