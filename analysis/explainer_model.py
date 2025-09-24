import json
import pandas as pd
from tqdm import tqdm
import os
import concurrent.futures
import random
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


from sentence_transformers import SentenceTransformer, util


# Parameters
K = 5  # Number of DIFFERENT explanations per neuron (changed from 5 to 20)
M = 5   # Number of exemplars to consider per neuron (both high and low)
LAYERS = 1
NUERONS = 1
MODEL = "microsoft/Phi-3.5-mini-instruct"

print("Loading Phi-3.5-mini-instruct model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    device_map="auto",
    load_in_8bit=True,    # or load_in_4bit=True
    attn_implementation="flash_attention_2",
    use_cache=True,   
)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded on device: {model.device}")

# The system prompt
with open('explainer_sys_prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read()

# Function to generate the user prompt with both high and low activations
def generate_prompt(neuron_id, neuron_top_list, neuron_bottom_list, dataset, previous_explanations=None, explanation_num=1):
    print(f"[DEBUG] Generating prompt for {neuron_id} with {len(neuron_top_list)} high and {len(neuron_bottom_list)} low exemplars...")
    
    def compress_features(feat_dict):
        keys_to_keep = [
        'Zinc finger', 'Coiled coil', 'Compositional bias',
        'Protein families', 'Domain [CC]', 'Modified residue', 'Glycosylation',
        'Lipidation', 'Signal peptide', 'Transit peptide', 'Disulfide bond',
        'Chain', 'Post-translational modification', 'Cross-link', 'Propeptide',
        'Peptide', 'Helix', 'Beta strand', 'Turn', 'Transmembrane',
        'Intramembrane', 'Topological domain', 'Subcellular location [CC]',
        'Gene Ontology (GO)', 'Gene Ontology (molecular function)',
        'Gene Ontology (biological process)',
        'Gene Ontology (cellular component)', 'Keywords', 'InterPro', 'Pfam',
        'SMART', 'PROSITE', 'PDB', 'AlphaFoldDB', 'CDD', 'HAMAP', 'PANTHER',
        'PIRSF', 'PRINTS', 'SUPFAM', 'Gene3D', 'alphafold_link', 'gene3d_info',
        'NonStandardAminoAcids', 'Amino Acid Percentages', 'Molecular Weight',
        'Aromaticity', 'Instability Index', 'GRAVY', 'Isoelectric Point',
        'Charge at pH 7', 'Helix Fraction'
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

    # High activation examples
    high_examples = "HIGH ACTIVATION EXAMPLES (what this neuron responds to):\n"
    for idx, item in enumerate(neuron_top_list):
        seq = item["sequence"]
        activation = item["activation"]
        print(f"[DEBUG] Looking up high sequence {idx+1}/{len(neuron_top_list)}: {seq[:15]}...")
        features = dataset[dataset['Sequence'] == seq].iloc[0].to_dict()
        compact = compress_features(features)
        high_examples += f"Seq: {seq}\nAct: {activation}\nFeat: {compact}\n\n"

    # Low activation examples (negative examples)
    low_examples = "LOW ACTIVATION EXAMPLES (what this neuron does NOT respond to):\n"
    for idx, item in enumerate(neuron_bottom_list):
        seq = item["sequence"]
        activation = item["activation"]
        print(f"[DEBUG] Looking up low sequence {idx+1}/{len(neuron_bottom_list)}: {seq[:15]}...")
        features = dataset[dataset['Sequence'] == seq].iloc[0].to_dict()
        compact = compress_features(features)
        low_examples += f"Seq: {seq}\nAct: {activation}\nFeat: {compact}\n\n"

    # Build the previous explanations section if this is not the first explanation
    previous_explanations_section = ""
    if previous_explanations and explanation_num > 1:
        previous_explanations_section = "PREVIOUS EXPLANATIONS (DO NOT REPEAT THESE):\n"
        for i, exp in enumerate(previous_explanations, 1):
            previous_explanations_section += f"{i}. {exp}\n"
        previous_explanations_section += "\n"

    return f"""Neuron: {neuron_id}

{high_examples.strip()}

{low_examples.strip()}

{previous_explanations_section}TASK: Write exactly ONE WORD or SHORT PHRASE that describes a DISTINCT biological theme that distinguishes high activation sequences from low activation sequences.

RULES:
- The answer must be only one word or short phrase (e.g., "Zinc-finger", "Sweet Membrane", "negative gravy scores", "transmembrane domains").
- Focus on what distinguishes the high activation examples from the low activation examples.
- Do not output sentences, punctuation, or dictionaries.
- Consider both what is present in high activation examples and absent in low activation examples.
- **CRITICAL: This is explanation #{explanation_num} of {K}. Your answer MUST be DIFFERENT from all previous explanations.**
- **Think of alternative interpretations, different aspects, or complementary biological functions.**
- **If you find yourself repeating a previous theme, consider a more specific sub-feature or a different perspective.**

Answer:"""


def call_megabeam(prompt, max_retries=3):
    """Generate response using MegaBeam-Mistral-7B-512k"""
    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] Calling MegaBeam (attempt {attempt+1})...")
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,   # let long context flow
                padding=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=25,     # short phrases only
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Keep only first line, trim junk
            response = response.splitlines()[0].strip()
            print(f"[DEBUG] Response: {repr(response[:200])}...")
            return response
        except Exception as e:
            print(f"[ERROR] attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return "Error generating explanation"


# Load embedder once globally
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def is_too_similar(new_exp, prev_vecs, threshold=0.8):
    """Check semantic similarity between new explanation and cached embeddings."""
    if len(prev_vecs) == 0:
        return False
    new_vec = embedder.encode(new_exp, convert_to_tensor=True)
    sims = util.cos_sim(new_vec, prev_vecs)
    return sims.max().item() >= threshold

def explain_single_neuron(layer, neuron, dataset, activations, M, K):
    neuron_id = f"layer_{layer}_neuron_{neuron}"
    print(f"[DEBUG] Explaining {neuron_id}...")

    # Get top/bottom activations
    try:
        top_k_high = list(activations[str(layer)]["top_k_high"][str(neuron)])[0:M]
    except (KeyError, IndexError):
        print(f"[WARN] No high activations found for {neuron_id}")
        top_k_high = []

    try:
        top_k_low = list(activations[str(layer)]["top_k_low"][str(neuron)])[0:M]
    except (KeyError, IndexError):
        print(f"[WARN] No low activations found for {neuron_id}")
        top_k_low = []

    if not top_k_high and not top_k_low:
        return {"neuron_id": neuron_id, **{f"explanation_{i+1}": "N/A" for i in range(K)}}

    row = {"neuron_id": neuron_id}
    previous_explanations = []
    previous_vecs = None  # cached embeddings

    for i in range(K):
        print(f"[DEBUG] Generating explanation {i+1}/{K} for {neuron_id}...")
        prompt = generate_prompt(neuron_id, top_k_high, top_k_low, dataset, previous_explanations, i+1)
        explanation_text = call_megabeam(prompt)
        cleaned_explanation = explanation_text.strip()

        # Check semantic similarity
        if previous_explanations:
            if is_too_similar(cleaned_explanation, previous_vecs):
                print(f"[WARN] Explanation {i+1} too similar to previous ones → retrying...")
                # Retry with more diversity
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=False,
                    padding=True
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=25,
                        temperature=0.9,  # encourage diversity
                        top_p=0.7,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                retry_response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                cleaned_explanation = retry_response.splitlines()[0].strip()
                print(f"[DEBUG] Retry response: {repr(cleaned_explanation[:200])}...")

        row[f"explanation_{i+1}"] = cleaned_explanation
        previous_explanations.append(cleaned_explanation)

        # Update embedding cache
        if previous_vecs is None:
            previous_vecs = embedder.encode([cleaned_explanation], convert_to_tensor=True)
        else:
            new_vec = embedder.encode([cleaned_explanation], convert_to_tensor=True)
            previous_vecs = torch.cat([previous_vecs, new_vec], dim=0)

    return row

def explain_neurons_parallel(dataset, activations, M, K, LAYERS, NUERONS, output_csv, max_workers=4):
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for layer in range(LAYERS):
            for neuron in range(NUERONS):
                futures.append(
                    executor.submit(explain_single_neuron, layer, neuron, dataset, activations, M, K)
                )

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="All neurons"):
            try:
                result = future.result()
                rows.append(result)
            except Exception as e:
                print(f"[ERROR] Future failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] CSV written to {output_csv}")


# Example usage
if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    dataset = pd.read_csv(
        "../swissprot_filtered_with_alphafold_link_and_cathgene3d_and_features.tsv.gz",
        sep="\t"
    )
    print("[INFO] Dataset loaded.")
    print("[INFO] Loading activations JSON...")
    activations = json.load(open("../datageneration/esm8M_finaldataset_k30_optimized.json"))
    print("[INFO] Activations loaded.")
    output_csv = "test.csv"
    explain_neurons_parallel(dataset, activations, M, K, LAYERS, NUERONS, output_csv, max_workers=4)