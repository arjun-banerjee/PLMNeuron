import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import os
import concurrent.futures
from openai._exceptions import APIStatusError
import random
import time



# Load your API key from .env.local
load_dotenv(".env.local")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Parameters
K = 2  # Number of explanations per neuron
M = 2 # Number of exemplars to consider per neuron
LAYERS = 6
NUERONS = 320
MODEL = "gpt-4o-mini"

# The system prompt
with open('explainer_sys_prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read()


# Function to generate the user prompt
def generate_prompt(neuron_id, neuron_top_list, dataset):
    def compress_features(feat_dict):
        keys_to_keep = [
            "length" "Mass", "mol_weight", "iso_point", "gravy", "charge_pH7",
            "helix_frac", "turn_frac", "sheet_frac", "instability_index",
            "boman_index", "aliphatic_index", "hydrophobic_moment",

            "Protein names", "Organism",
            "Subcellular location [CC]",
            "Gene Ontology (biological process)",
            "Gene Ontology (molecular function)",
            "Function [CC]", "Disruption phenotype",
            "Catalytic activity", "Pathway"
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

    examples = ""
    for item in neuron_top_list:
        seq = item["sequence"]
        activation = item["activation"]
        features = dataset[dataset['Sequence'] == seq].iloc[0].to_dict()
        compact = compress_features(features)
        examples += f"Seq: {seq}\nAct: {activation}\nFeat: {compact}\n\n"

    return f"""Neuron: {neuron_id}
The following are sequences and their associated biological features where the neuron strongly activates. Summarize the shared biological features in one sentence using the fewest words possible.
{examples.strip()}"""


def call_openai(prompt, max_retries=10):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Your system prompt here"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        except APIStatusError as e:
            if getattr(e, "status_code", None) == 429:
                wait_time = random.uniform(1, 2) * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {wait_time:.2f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise  # Re-raise non-rate-limit errors

        except Exception as e:
            print(f"Unhandled exception on attempt {attempt + 1}: {e}")
            time.sleep(2)

    raise RuntimeError("Max retries exceeded for OpenAI call.")



"""
def explain_neurons(dataset, activations, M, K, LAYERS, NUERONS, output_csv):
    rows = []
    for layer in tqdm(range(LAYERS), desc="Going through layer"):
        for neuron in tqdm(range(NUERONS), desc="Explaining neurons"):
            neuron_id = f"layer_{layer}_neuron_{neuron}"
            top_k_high = list(activations[str(layer)]["top_k_high"][str(neuron)])[0:M]
            row = {"neuron_id": neuron_id}  
            for i in range(K):
                prompt = generate_prompt(neuron_id, top_k_high, dataset)
                explanation_text = call_openai(prompt)
                row[f"explanation_{i+1}"] = explanation_text.strip()
            print(row)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV written to {output_csv}")
"""

def explain_single_neuron(layer, neuron, dataset, activations, M, K):
    neuron_id = f"layer_{layer}_neuron_{neuron}"
    try:
        top_k_high = list(activations[str(layer)]["top_k_high"][str(neuron)])[0:M]
    except (KeyError, IndexError):
        return {"neuron_id": neuron_id, **{f"explanation_{i+1}": "N/A" for i in range(K)}}

    row = {"neuron_id": neuron_id}
    for i in range(K):
        prompt = generate_prompt(neuron_id, top_k_high, dataset)
        explanation_text = call_openai(prompt)
        #print(prompt)
        row[f"explanation_{i+1}"] = explanation_text.strip()
    return row


def explain_neurons_parallel(dataset, activations, M, K, LAYERS, NUERONS, output_csv):
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
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
                print(f"Error: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV written to {output_csv}")


# Example usage
if __name__ == "__main__":
    dataset = pd.read_parquet("../datatest573230.parquet", engine="pyarrow")
    actvations = json.load(open("../esm8M_500kdataset_k100_optimized.json"))
    output_csv = "esm8M_500k_neuron_explanations.csv"

    explain_neurons_parallel(dataset, actvations, M, K, LAYERS, NUERONS, output_csv)