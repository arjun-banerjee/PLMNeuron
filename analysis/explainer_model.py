import json
import pandas as pd
from tqdm import tqdm
import os
import concurrent.futures
import random
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gc
import threading
from collections import defaultdict

from sentence_transformers import SentenceTransformer, util

# Parameters
K = 5  # Number of DIFFERENT explanations per neuron
M = 4   # Number of exemplars to consider per neuron (both high and low)
LAYERS = 6
NUERONS = 320
MODEL = "microsoft/Phi-3.5-mini-instruct"

print("Loading Phi-3.5-mini-instruct model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)


# Improved model loading with better memory management
model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    use_cache=False,
    low_cpu_mem_usage=True,
    offload_folder="offload",
    offload_state_dict=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded on device: {model.device}")

# Load system prompt
with open('explainer_sys_prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read()

# Load embedder once globally
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def compress_features(feat_dict):
    keys_to_keep = [
        # --- Basic identifiers / metadata ---
        'Gene Names', 'Organism', 'Protein names',
        'Length', 'Mass', 'Sequence',

        # --- Functional annotations ---
        'Active site', 'Binding site', 'Non-standard residue', 'Site',
        'DNA binding', 'Catalytic activity', 'Function [CC]', 'EC number',
        'Pathway', 'Cofactor', 'Activity regulation',

        # --- Structural features ---
        'Domain [FT]', 'Motif', 'Region', 'Repeat',
        'Zinc finger', 'Coiled coil', 'Compositional bias',
        'Protein families', 'Modified residue',
        'Glycosylation', 'Lipidation', 'Signal peptide', 'Transit peptide',
        'Disulfide bond', 'Chain', 'Post-translational modification',
        'Cross-link', 'Propeptide', 'Peptide', 'Helix', 'Beta strand',
        'Turn', 'Transmembrane', 'Intramembrane', 'Topological domain',
        'Subcellular location [CC]',

        # --- Ontologies / databases ---
        'Gene Ontology (GO)', 'Gene Ontology (molecular function)',
        'Gene Ontology (biological process)', 'Gene Ontology (cellular component)',
        'Keywords', 'gene3d_info',

        # --- Special sequence features ---
        'NonStandardAminoAcids',

        # --- Computed peptide features (peptides.Peptide) ---
        'Aliphatic Index', 'Atchley Factors', 'Blosum Indices', 'Boman Index',
        'Cruciani Properties', 'Akashi Energy Cost', 'Craig Energy Cost',
        'Heizer Energy Cost', 'Wagner Energy Cost', 'Hydrophobic Moment',
        'Mass over Charge Ratio', 'Nutrient Cost', 'Physical Chemical Properties',
        'PRIN Components', 'Protfp Descriptors', 'Sneath Vectors', 'ST Scales',
        'Structural Class',

        # --- Computed protein features (ProteinAnalysis) ---
        'Amino Acid Percentages', 'Molecular Weight', 'Aromaticity',
        'Instability Index', 'GRAVY', 'Isoelectric Point', 'Charge at pH 7',
        'Helix Fraction', 'Turn Fraction', 'Sheet Fraction',
        'Molecular Extinction Coefficient'
    ]

    compressed = {}
    for key in keys_to_keep:
        val = feat_dict.get(key, "nan")
        if isinstance(val, float):
            val = round(val, 4)
        if isinstance(val, str) and val.lower() == "nan":
            continue
        compressed[key] = val
    return compressed

class DatasetLookup:
    """Optimized dataset lookup with caching"""
    def __init__(self, dataset):
        print("[INFO] Building sequence lookup cache...")
        self.sequence_to_features = {}
        for idx, row in dataset.iterrows():
            self.sequence_to_features[row['Sequence']] = row.to_dict()
        print(f"[INFO] Cached {len(self.sequence_to_features)} sequences")
    
    def get_features(self, sequence):
        return self.sequence_to_features.get(sequence, {})

def generate_prompt_batch(neuron_ids, neuron_data_list, dataset_lookup, previous_explanations_list=None, explanation_nums=None):
    """Generate multiple prompts at once"""
    prompts = []
    
    for i, (neuron_id, (neuron_top_list, neuron_bottom_list)) in enumerate(zip(neuron_ids, neuron_data_list)):
        # High activation examples
        high_examples = "HIGH ACTIVATION EXAMPLES:\n"
        for item in neuron_top_list:
            activation = round(item["activation"], 4)
            features = dataset_lookup.get_features(item["sequence"])
            compact = compress_features(features)
            high_examples += f"Act: {activation}\nFeat: {str(compact)}\n\n"

        # Low activation examples
        low_examples = "LOW ACTIVATION EXAMPLES:\n"
        for item in neuron_bottom_list:
            activation = round(item["activation"], 4)
            features = dataset_lookup.get_features(item["sequence"])
            compact = compress_features(features)
            low_examples += f"Act: {activation}\nFeat: {str(compact)}\n\n"

        # Previous explanations section
        previous_explanations_section = ""
        if previous_explanations_list and i < len(previous_explanations_list) and previous_explanations_list[i]:
            previous_explanations_section = "PREVIOUS EXPLANATIONS (DO NOT REPEAT):\n"
            for j, exp in enumerate(previous_explanations_list[i], 1):
                previous_explanations_section += f"{j}. {exp}\n"
            previous_explanations_section += "\n"

        prompt =  f"""Neuron: {neuron_id}

{high_examples.strip()}

{low_examples.strip()}

{previous_explanations_section}TASK: Write exactly ONE WORD or SHORT PHRASE that describes a DISTINCT biological theme that distinguishes high activation sequences from low activation sequences.

RULES:
- The answer must be only one word or short phrase (e.g., "Zinc-finger", "Sweet Membrane", "negative gravy scores", "transmembrane domains").
- Focus on what distinguishes the high activation examples from the low activation examples.
- Do not output sentences, punctuation, or dictionaries.
- Consider both what is present in high activation examples and absent in low activation examples.
- **CRITICAL: This is explanation #{explanation_nums} of {K}. Your answer MUST be DIFFERENT from all previous explanations.**
- **Think of alternative interpretations, different aspects, or complementary biological functions.**
- **If you find yourself repeating a previous theme, consider a more specific sub-feature or a different perspective.**

Answer:"""
        
        prompts.append(prompt)
    
    return prompts

def call_model_batch(prompts, max_retries=1, batch_size=32):
    """Generate responses for multiple prompts with batching"""
    all_responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        for attempt in range(max_retries):
            try:
                torch.cuda.empty_cache()
                
                # Tokenize batch
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=2048  # Limit context length
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,  # Reduced from 100
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Decode responses
                batch_responses = []
                for j, output in enumerate(outputs):
                    gen_tokens = output[inputs['input_ids'].shape[1]:]
                    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    batch_responses.append(response.splitlines()[0].strip())

                all_responses.extend(batch_responses)
                
                # Clean up
                del inputs, outputs
                torch.cuda.empty_cache()
                break
                
            except Exception as e:
                print(f"[ERROR] Batch attempt {attempt+1} failed: {e}")
                torch.cuda.empty_cache()
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    all_responses.extend(["Error generating explanation"] * len(batch_prompts))
    
    return all_responses

# def is_too_similar_batch(new_explanations, previous_vecs_list, threshold=0.9):
#     """Check similarity for batch of explanations"""
#     if not new_explanations:
#         return [False] * len(new_explanations)
    
#     # Encode all new explanations at once
#     new_vecs = embedder.encode(new_explanations, convert_to_tensor=True)
#     similarities = []
    
#     for i, (new_vec, prev_vecs) in enumerate(zip(new_vecs, previous_vecs_list)):
#         if prev_vecs is None or len(prev_vecs) == 0:
#             similarities.append(False)
#         else:
#             sims = util.cos_sim(new_vec.unsqueeze(0), prev_vecs)
#             similarities.append(sims.max().item() >= threshold)
    
#     return similarities

def process_neuron_batch(batch_data, dataset_lookup, M, K):
    """Process a batch of neurons efficiently"""
    neuron_ids, layer_neuron_pairs, activations = batch_data
    
    # Prepare neuron data
    neuron_data_list = []
    for layer, neuron in layer_neuron_pairs:
        try:
            top_k_high = list(activations[str(layer)]["top_k_high"][str(neuron)])[0:M]
        except (KeyError, IndexError):
            top_k_high = []

        try:
            top_k_low = list(activations[str(layer)]["top_k_low"][str(neuron)])[0:M]
        except (KeyError, IndexError):
            top_k_low = []
            
        neuron_data_list.append((top_k_high, top_k_low))
    
    # Initialize results
    results = []
    for neuron_id in neuron_ids:
        results.append({
            "neuron_id": neuron_id,
            **{f"explanation_{i+1}": "" for i in range(K)}
        })
    
    # Process each explanation round
    previous_explanations_list = [[] for _ in neuron_ids]
    previous_vecs_list = [None for _ in neuron_ids]
    
    for explanation_round in range(K):
        # Generate prompts for this round
        prompts = generate_prompt_batch(
            neuron_ids, 
            neuron_data_list, 
            dataset_lookup,
            previous_explanations_list,
            [explanation_round + 1] * len(neuron_ids)
        )
        
        # Get responses
        responses = call_model_batch(prompts)
        
        # Check similarities
        #similarities = is_too_similar_batch(responses, previous_vecs_list)
        
        # Handle similar responses with retries
        # retry_indices = [i for i, is_similar in enumerate(similarities) if is_similar]
        # if retry_indices:
        #     retry_prompts = [prompts[i] + "\n\nBe more creative and specific:" for i in retry_indices]
        #     retry_responses = call_model_batch(retry_prompts)
        #     for i, retry_idx in enumerate(retry_indices):
        #         responses[retry_idx] = retry_responses[i]
        
        # Update results and caches
        for i, (response, neuron_id) in enumerate(zip(responses, neuron_ids)):
            cleaned_response = response.strip()
            results[i][f"explanation_{explanation_round + 1}"] = cleaned_response
            previous_explanations_list[i].append(cleaned_response)
            
            # Update embedding cache
            if previous_vecs_list[i] is None:
                previous_vecs_list[i] = embedder.encode([cleaned_response], convert_to_tensor=True)
            else:
                new_vec = embedder.encode([cleaned_response], convert_to_tensor=True)
                previous_vecs_list[i] = torch.cat([previous_vecs_list[i], new_vec], dim=0)
    
    return results

def explain_neurons_optimized(dataset, activations, M, K, LAYERS, NUERONS, output_csv, batch_size=16):
    """Optimized neuron explanation with batching"""
    
    # Create dataset lookup
    dataset_lookup = DatasetLookup(dataset)
    
    # Prepare all neuron data
    all_neuron_data = []
    for layer in range(LAYERS):
        for neuron in range(NUERONS):
            neuron_id = f"layer_{layer}_neuron_{neuron}"
            all_neuron_data.append((neuron_id, (layer, neuron)))
    
    total_neurons = len(all_neuron_data)
    all_results = []
    
    # Process in batches
    with tqdm(total=total_neurons, desc="Processing neurons") as pbar:
        for i in range(0, total_neurons, batch_size):
            batch_data_raw = all_neuron_data[i:i+batch_size]
            
            # Prepare batch
            neuron_ids = [data[0] for data in batch_data_raw]
            layer_neuron_pairs = [data[1] for data in batch_data_raw]
            batch_data = (neuron_ids, layer_neuron_pairs, activations)
            
            try:
                batch_results = process_neuron_batch(batch_data, dataset_lookup, M, K)
                all_results.extend(batch_results)
                
                # Periodic cleanup and saving
                if len(all_results) % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Save intermediate results
                    df_temp = pd.DataFrame(all_results)
                    df_temp.to_csv(f"{output_csv}.temp", index=False)
                    print(f"[INFO] Processed {len(all_results)} neurons, saved checkpoint")
                
                pbar.update(len(batch_data_raw))
                
            except Exception as e:
                print(f"[ERROR] Failed to process batch starting at index {i}: {e}")
                
                # Add error rows for the batch
                for neuron_id, _ in batch_data_raw:
                    error_row = {"neuron_id": neuron_id}
                    error_row.update({f"explanation_{j+1}": "ERROR" for j in range(K)})
                    all_results.append(error_row)
                
                pbar.update(len(batch_data_raw))
                torch.cuda.empty_cache()
                gc.collect()

    # Save final results
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    
    # Clean up temp file
    temp_file = f"{output_csv}.temp"
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    print(f"[INFO] Final CSV written to {output_csv}")

# Example usage
if __name__ == "__main__":
    # Set memory optimization environment variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print("[INFO] Loading dataset...")
    dataset = pd.read_parquet(
        "../datageneration/swiss_prot.parquet",
        engine="pyarrow"
    )

    print("[INFO] Dataset loaded.")
    print("[INFO] Loading activations JSON...")
    activations = json.load(open("../datageneration/esm8M_finaldataset_k30_optimized.json"))
    print("[INFO] Activations loaded.")
    
    output_csv = "ESM8MExplinations_Optimized.csv"
    explain_neurons_optimized(dataset, activations, M, K, LAYERS, NUERONS, output_csv, batch_size=32)