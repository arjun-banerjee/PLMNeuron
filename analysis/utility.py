import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random
import csv
import re
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os

# Configuration
CSV_PATH = "esm35M_500k_neuron_explanations.csv"
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
SEQ_LEN = 300
NUM_STEPS = 200
A = 5
B = 3

# Utilities
def find_matching_neurons(csv_path, keyword):
    matches = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                continue
            neuron_id = row[0]
            if any(keyword in desc.lower() for desc in row[1:]):
                m = re.match(r"layer_(\d+)_neuron_(\d+)", neuron_id)
                if m:
                    matches.append((int(m.group(1)), int(m.group(2))))
    return matches

def make_multi_neuron_hook(neurons, a=10.0, b=3.0):
    def hook_fn(module, input, output):
        for neuron in neurons:
            output[:, :, neuron] = a * output[:, :, neuron] + b
        return output
    return hook_fn

def random_protein_sequence(length):
    return ''.join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length))

def sample_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def decode_tokens(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")

def compute_gravy_ignore_x(seq):
    cleaned = seq.replace("X", "")
    if not cleaned:
        return None
    return ProteinAnalysis(cleaned).instability_index()

def compute_gravy_score(seq):
    cleaned = seq.replace("X", "")
    if not cleaned:
        return None
    return ProteinAnalysis(cleaned).gravy()

# Main steering function
def steer(model, tokenizer, base_sequence, match_string, label, compute_metric_func=compute_gravy_ignore_x):
    matched_neurons = find_matching_neurons(CSV_PATH, match_string)
    if not matched_neurons:
        print(f"No matching neurons found for: {match_string}")
        return []

    layer_to_neurons = defaultdict(list)
    for layer, neuron in matched_neurons:
        layer_to_neurons[layer].append(neuron)

    device = model.device
    history = []  # (step, sequence, activation, metric_value)

    seq = base_sequence
    for step in tqdm(range(NUM_STEPS), desc=f"Steering for {label}"):
        inputs = tokenizer(seq, return_tensors="pt").to(device)

        handles = []
        for layer, neurons in layer_to_neurons.items():
            hook = model.base_model.encoder.layer[layer].intermediate.register_forward_hook(
                make_multi_neuron_hook(neurons, A, B)
            )
            handles.append(hook)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0]
            hidden_states = outputs.hidden_states

        for h in handles:
            h.remove()

        # Score: average absolute activation across target neurons
        total, count = 0, 0
        for layer, neurons in layer_to_neurons.items():
            h = hidden_states[layer][0]
            for n in neurons:
                total += abs(h[:, n].mean().item())
                count += 1
        avg_act = total / count if count > 0 else 0

        sampled_ids = sample_from_logits(logits)
        seq = decode_tokens(tokenizer, sampled_ids)
        metric_value = compute_metric_func(seq)
        history.append((step + 1, seq, avg_act, metric_value, label))

    return history

def run_multiple_steering_experiments(steering_configs, csv_output_path):
    """
    Run multiple steering experiments and generate separate plots for each pair.
    
    Args:
        steering_configs: List of dictionaries, each containing:
            - 'pos_match': positive match string
            - 'neg_match': negative match string  
            - 'compute_metric_func': function to compute metric from sequence
            - 'plot_title': title for the plot
            - 'y_label': y-axis label for the plot
        csv_output_path: Path to save the CSV results
    """
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Generate shared initial sequence
    base_sequence = random_protein_sequence(SEQ_LEN)
    print("Starting sequence:", base_sequence)

    all_results = []
    
    for i, config in enumerate(steering_configs):
        pos_match = config['pos_match']
        neg_match = config['neg_match']
        compute_metric_func = config['compute_metric_func']
        plot_title = config['plot_title']
        y_label = config['y_label']
        
        print(f"\nRunning experiment {i+1}/{len(steering_configs)}: {pos_match} vs {neg_match}")
        
        # Run both steering directions
        history_pos = steer(model, tokenizer, base_sequence, match_string=pos_match, label=f"pos_{i}", compute_metric_func=compute_metric_func)
        history_neg = steer(model, tokenizer, base_sequence, match_string=neg_match, label=f"neg_{i}", compute_metric_func=compute_metric_func)
        
        # Add initial point (step 0) before steering
        initial_metric = compute_metric_func(base_sequence)
        init_row_pos = (0, base_sequence, 0.0, initial_metric, f"pos_{i}")
        init_row_neg = (0, base_sequence, 0.0, initial_metric, f"neg_{i}")
        history_pos = [init_row_pos] + history_pos
        history_neg = [init_row_neg] + history_neg
        
        # Store results with experiment info
        for row in history_pos + history_neg:
            step, seq, act, metric, label = row
            # Add experiment metadata
            experiment_id = i
            pos_match_str = pos_match
            neg_match_str = neg_match
            all_results.append([experiment_id, step, seq, act, metric, label, pos_match_str, neg_match_str])
        
        # Create individual plot for this experiment
        plot_data = history_pos + history_neg
        df = pd.DataFrame(plot_data)
        df.columns = ["step", "sequence", "activation", "metric_value", "label"]
        
        sns.set(style="whitegrid", font_scale=1.2)
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="step", y="metric_value", hue="label", marker="o")
        plt.xlabel("Step")
        plt.ylabel(y_label)
        plt.title(f"{plot_title}: Experiment {i+1} ({pos_match} vs {neg_match})")
        plt.legend(title="Steering Direction")
        plt.tight_layout()
        
        # Save individual plot
        plot_filename = f"steering_experiment_{i+1}.png"
        plt.savefig(plot_filename)
        plt.show()
        plt.close()

    # Save all results to CSV
    with open(csv_output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_id", "step", "sequence", "activation", "metric_value", "label", 
                        "pos_match_string", "neg_match_string"])
        for row in all_results:
            writer.writerow(row)
    
    print(f"\nAll results saved to: {csv_output_path}")
    print(f"Individual plots saved as: steering_experiment_1.png, steering_experiment_2.png, etc.")

# Example usage function
def run_experiments():
    """Example function showing how to use the new scalable system."""
    
    steering_configs = [
        {
            'pos_match': "high instability indices",
            'neg_match': "low instability indices",
            'compute_metric_func': compute_gravy_ignore_x,
            'plot_title': "Instability Index Score Trajectories",
            'y_label': "Instability Index Score"
        },
        {
            'pos_match': "positive gravy score neurons",
            'neg_match': "negative gravy score neurons",
            'compute_metric_func': compute_gravy_score,
            'plot_title': "GRAVY Score Trajectories",
            'y_label': "GRAVY Score"
        }
    ]
    
    run_multiple_steering_experiments(
        steering_configs=steering_configs,
        csv_output_path="steered_sequences_multiple.csv"
    )

# Run both steering loops (original functionality preserved)
if __name__ == "__main__":
    # Example usage of the new scalable system
    run_experiments()
