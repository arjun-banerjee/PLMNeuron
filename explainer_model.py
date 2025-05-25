import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load your API key from .env.local
load_dotenv(".env.local")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Parameters
K = 2  # Number of explanations per neuron
MODEL = "gpt-4o"

# The system prompt
def load_prompt(path="system_prompt.txt"):
    with open(path, "r") as f:
        return f.read()

SYSTEM_PROMPT = load_prompt()

# Prompt generator
def generate_prompt(neuron_id, top_k_high):
    examples = ""
    for item in top_k_high:
        seq = item["sequence"]
        activation = item["activation"]
        features_str = f"Features: {item.get('features', '')}" if 'features' in item else ""
        examples += f"Sequence: {seq}\nActivation: {activation}\n{features_str}\n\n"
    
    return f"""Neuron ID: {neuron_id}
Below are the top activating sequences and their features for this neuron. Describe the biological features that cause the neuron to activate.

{examples}"""

def call_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def explain_neurons(neuron_data: dict, output_csv: str, k: int = K):
    rows = []

    for neuron_id, content in tqdm(neuron_data.items(), desc="Explaining neurons"):
        top_k_high = content.get("top_k_high", [])
        row = {"neuron_id": neuron_id}

        for i in range(k):
            prompt = generate_prompt(neuron_id, top_k_high)
            
            # Uncomment this for real inference:
            explanation_text = call_openai(prompt)

            # Placeholder:
            # explanation_text = f"Dummy explanation {i+1} for neuron {neuron_id}"

            row[f"explanation_{i+1}"] = explanation_text.strip()

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV written to {output_csv}")


# Example usage
if __name__ == "__main__":
    with open("neuron_activations.json") as f:
        data = json.load(f)
    
    explain_neurons(data, "neuron_explanations.csv", k=K)
