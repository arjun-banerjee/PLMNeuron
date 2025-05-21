import json
import heapq
import torch
from nnsight import LanguageModel
from transformers import ESMTokenizer, ESMModel

#Configurations
ESM_NAME = "facebook/esm2_t6_8b_512"
BATCH_SIZE = 16
K = 150

#Dataset
#TODO: @ETHAN + @DAVID GRAB DATA


#1 -- Load tokenizer and base model without LM head
tokenizer = ESMTokenizer.from_pretrained(ESM_NAME)
model = ESMModel.from_pretrained(ESM_NAME)

#2 -- Wrap with NNSight for tracing
lm = LanguageModel(esm_model, tokenizer=tokenizer, device_map="auto")

#3 -- Inspect archetecture
hf = lm.model
layers = hf.encoder.layers
n_layers = len(layers)
H = hf.config.hidden_size


#4 -- Heaps -- 1 per layer x neuron
heaps = [
    {high: [[] for _ in range(H)],
     low: [[] for _ in range(H)]}
    for _ in range(n_layers)
]

#5 -- stream through mini batches

for start in range(0, len(dataset), BATCH_SIZE):
    #batch_seq = #TODO: @ETHAN + @DAVID GRAB DATA
    inputs = tokenizer(batch_seq, return_tensors="pt", padding=True, truncation=True)
    batch_ids = inputs["input_ids"].to(device)
    
    #Trace this minibatch
    with lm.trace(batch_ids) as tracer:
        #save activations
        saved = []
        for l in layers:
            saved.append(l.output.save())
    
    for layer_idx, tensor in enumerate(saved):
        #[B, L, H] -> max over tokens -> [B, H]
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

# 6=--  After streaming all batches, extract and save exemplars
results = {}
for l in range(n_layers):
    results[l] = {"top_k_high": {}, "top_k_low": {}}
    for h in range(H):
        high_list = heapq.nlargest(K, heaps[l]["high"][h])
        low_list  = heapq.nlargest(K, heaps[l]["low"][h])
        
        #TODO: @ETHAN + @DAVID GRAB DATA 
        results[l]["top_k_high"][h] = [
            {"sequence": dataset[idx], "activation": score}
            for score, idx in high_list
        ]
        results[l]["top_k_low"][h] = [
            {"sequence": dataset[idx], "activation": -score}
            for score, idx in low_list
        ]

with open("esm2_neuron_exemplars.json", "w") as f:
    json.dump(results, f, indent=2)
      
                    
                


        
        