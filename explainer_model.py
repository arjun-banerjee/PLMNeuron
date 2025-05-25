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
SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into a specific neuron inside the esm2_t6_8M_UR50D language model that activates in response to text excerpts. Your overall task is to describe the biological features of protein sequences that cause the neuron to strongly activate.

You will receive a list of protein sequences on which the neuron activates. Tokens causing activation will appear between delimiters like {{this}}. Consecutive activating tokens will also be accordingly delimited {{just like this}}. If no tokens are highlighted with {{}}, then the neuron does not activate on any tokens in the excerpt. You will also receive a list of quantitative and qualitative biological features associated with the protein sequence, which will be notated by “Features:”. Additionally, the activation value for each sequence will be provided as a float, notated by “Activation:”.

Note: Neurons activate on an amino acid-by-amino acid basis. Also, neuron activations can only depend on amino acids before the amino acid it activates on, so the description cannot depend on amino acids that come after, and should only depend on amino acids that come before the activation.

Note: make your final descriptions as concise as possible, using as few words as possible to describe protein sequence features that activate the neuron."""

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
