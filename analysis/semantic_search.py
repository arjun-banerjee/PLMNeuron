from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import re

# Load your API key from .env.local
load_dotenv(".env.local")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def semantic_search(neuron_label, characteristic):
    prompt = """
    Answer only with a True or False. A neuron described as {neuron} be useful in trying
    to generate a protein with the following characteristic: {characteristic}?
    For example:
    Prompt: "Answer only with a True or False. A neuron described as "Encodes information
    about Zinc Fingers" be useful in trying to generate a protein with the following
    characteristic: "Alpha-Sheet"?", Answer: False
    Prompt: "Answer only with a True or False. A neuron described as "Associated with high
    hydrophobicity" be useful in trying to generate a protein with the following
    characteristic: "Increasing hydrophobicity"?", Answer: True
    """
    # call gpt
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Neuron: {neuron_label}, Characteristic: {characteristic}"}
        ],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    if content:
        return content.strip()

def search_all_neurons(characteristic, neuron_data_path, neuron_explanation_col_name):
    """
    Returns all neurons that semantic_search says matches the characteristic.
    Output: list of (layer, neuron) tuples.
    """
    data = pd.read_csv(neuron_data_path)
    matches = []
    for idx, row in data.iterrows():
        neuron_id = row["neuron_id"]
        neuron_label = row[neuron_explanation_col_name]
        # Call the LLM to check if this neuron matches the characteristic
        result = semantic_search(neuron_label, characteristic)
        if result and result.strip().lower() == "true":
            m = re.match(r"layer_(\d+)_neuron_(\d+)", str(neuron_id))
            if m:
                matches.append((int(m.group(1)), int(m.group(2))))
    return matches

if __name__ == "__main__":
    matches = search_all_neurons("Increasing hydrophobicity", "../esm35M_500k_neuron_explanations.csv", "explanation_1")
    print(matches)