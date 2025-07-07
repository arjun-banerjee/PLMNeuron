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
# CSV_PATH = "esm8M_500k_neuron_explanations.csv"
# CSV_PATH = "esm35M_500k_neuron_explanations.csv"
CSV_PATH = "esm3B_500k_neuron_explanations.csv"
MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
SEQ_LEN = 100
NUM_STEPS = 150
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

def compute_instability_index(seq):
    cleaned = seq.replace("X", "")
    if not cleaned:
        return None
    return ProteinAnalysis(cleaned).instability_index()

def compute_charge_at_ph7(seq):
    """Compute charge at pH 7 with intelligent fallback replacements."""
    cleaned_seq = clean_sequence(seq)
    if not cleaned_seq:
        return None
    try:
        return ProteinAnalysis(cleaned_seq).charge_at_pH(7.0)
    except Exception as e:
        print(f"Error computing charge at pH 7: {e}")
        return None


REPLACEMENTS = {
    "X": "A",
    "B": "D",
    "Z": "E",
    "U": "C",
    "O": "K"
}

def clean_sequence(seq):
    return ''.join(REPLACEMENTS.get(aa, aa) for aa in seq)

def compute_gravy_score(seq):
    """Compute GRAVY score with intelligent fallback replacements."""
    cleaned_seq = clean_sequence(seq)
    if not cleaned_seq or not cleaned_seq.isalpha():
        return None
    try:
        return ProteinAnalysis(cleaned_seq).gravy()
    except Exception as e:
        print(f"Error computing GRAVY: {e}")
        return None

def compute_molecular_weight(seq):
    """Compute molecular weight."""
    cleaned = seq.replace("X", "")
    if not cleaned:
        return None
    return ProteinAnalysis(cleaned).molecular_weight()


# Main steering function
def steer(model, tokenizer, base_sequence, match_string, label, compute_metric_func, use_random_neurons=False, num_random_neurons=50, a=5.0, b=3.0):
    if use_random_neurons:
        # Use random neurons as control
        matched_neurons = []
        # Get total number of layers and neurons from the model
        num_layers = len(model.base_model.encoder.layer)
        #num_neurons = model.base_model.encoder.layer[0].intermediate.dense.out_features
        hidden_size = model.config.hidden_size
        
        # Select num_random_neurons random neurons from random layers
        for _ in range(num_random_neurons):
            layer = random.randint(0, num_layers - 1)
            #neuron = random.randint(0, num_neurons - 1)
            neuron = random.randint(0, hidden_size - 1)
            matched_neurons.append((layer, neuron))
        
        print(f"Using {len(matched_neurons)} random neurons as control")
    else:
        # Use matched neurons
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
                make_multi_neuron_hook(neurons, a, b)
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
    banned_symbols = {'U', 'B', 'Z', 'X', 'O'} 
    max_restarts = 10000
    base_sequence = random_protein_sequence(SEQ_LEN)

    while any(sym in str(base_sequence) for sym in banned_symbols):
        if max_restarts == 0:
            print("Too many sequence retries")
            exit(1)
        base_sequence = random_protein_sequence(SEQ_LEN)
        max_restarts -= 1

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
        a = config.get('a', 5.0)
        b = config.get('b', 3.0)
        num_random_neurons = config.get('num_random_neurons', 50)

        new_labels = {
            0: ("Low Instability", "High Instability"),
            1: ("Positive GRAVY", "Negative GRAVY"),
            2: ("High Mol. Weight", "Low Mol. Weight"),
            3: ("Random Neurons A", "Random Neurons B"),
        }

        pos_label, neg_label = new_labels.get(i, (f"pos_{i}", f"neg_{i}"))
        
        history_pos = steer(model, tokenizer, base_sequence, match_string=pos_match, label=pos_label, 
                          compute_metric_func=compute_metric_func, use_random_neurons=config.get('use_random_neurons', False),
                          num_random_neurons=num_random_neurons, a=a, b=b)
        history_neg = steer(model, tokenizer, base_sequence, match_string=neg_match, label=neg_label, 
                          compute_metric_func=compute_metric_func, use_random_neurons=config.get('use_random_neurons', False),
                          num_random_neurons=num_random_neurons, a=a, b=b)
        
        # Add initial point (step 0) before steering
        initial_metric = compute_metric_func(base_sequence)
        init_row_pos = (0, base_sequence, 0.0, initial_metric, pos_label)
        init_row_neg = (0, base_sequence, 0.0, initial_metric, neg_label)
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

    df_all = pd.DataFrame(all_results, columns=[
        "experiment_id", "step", "sequence", "activation", "metric_value",
        "label", "pos_match_string", "neg_match_string"
    ])
    df_all.to_csv(csv_output_path, index=False)
    print(f"\nAll results saved to: {csv_output_path}")

    custom_titles = {
    0: "Low vs High Instability",
    1: "Positive vs Negative GRAVY",
    2: "High vs Low Molecular Weight",
    3: "Random Neurons"
}
    custom_axis = {
        0: "Instability Index",
        1: "GRAVY",
        2: "Molecular Weight",
        3: "GRAVY"
    }

    # Create the FacetGrid
    sns.set(style="whitegrid", font_scale=1.1)
    g = sns.FacetGrid(
        df_all,
        col="experiment_id",
        hue="label",
        col_wrap=2,
        sharey=False,
        height=4,
        aspect=1.5,
        margin_titles=True
    )
    g.map(sns.lineplot, "step", "metric_value", marker="o")
    g.add_legend(title="Label")

    # Manually set titles
    for ax, title_key in zip(g.axes.flat, g.col_names):
        ax.set_title(custom_titles[int(title_key)], fontsize=13)
        ax.set_ylabel(custom_axis[int(title_key)], fontsize=13)

    # Adjust layout
    plt.subplots_adjust(top=0.9, right=0.85)
    g._legend.set_bbox_to_anchor((1, 0.5))
    g._legend.set_frame_on(True)
    g.fig.suptitle("ESM2-3B Single Characteristic Steering", fontsize=16)

    # Save plot
    plt.savefig("3B_combined_steering_experiments_custom_titles.png", bbox_inches="tight")
    plt.close()
    print("Saved with custom experiment titles.")

    # converged seuqences

    df_weight = df_all[df_all["experiment_id"] == 2]
    last_step = df_weight["step"].max()
    converged = df_weight[df_weight["step"] == last_step]

    print("\nConverged sequences:")
    for _, row in converged.iterrows():
        label = row["label"]
        seq = row["sequence"]
        metric = row["metric_value"]
        print(f"[{label}] Final Metric: {metric:.2f}")
        print("Length: " + str(len(seq)))
        print(seq)
        print("Amino Acid Frequencies:")
        freqs = pd.Series(list(seq)).value_counts(normalize=True).sort_index()
        for aa, f in freqs.items():
            print(f"  {aa}: {f:.3f}")
        print("-" * 50)

    # # Save all results to CSV
    # with open(csv_output_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["experiment_id", "step", "sequence", "activation", "metric_value", "label", 
    #                     "pos_match_string", "neg_match_string"])
    #     for row in all_results:
    #         writer.writerow(row)
    
    # print(f"\nAll results saved to: {csv_output_path}")
    # print(f"Individual plots saved as: steering_experiment_1.png, steering_experiment_2.png, etc.")

# Example usage function
def run_experiments():
    """Example function showing how to use the new scalable system."""
    
    steering_configs = [
        {
            'pos_match': "high instability",
            'neg_match': "low instability",
            'compute_metric_func': compute_instability_index,
            'plot_title': "Instability Index Score Trajectories",
            'y_label': "Instability Index Score",
            'use_random_neurons': False,  # Use matched neurons
            'a': 5.0,
            'b': 3.0
        },
        {
            'pos_match': "positive gravy",
            'neg_match': "negative gravy",
            'compute_metric_func': compute_gravy_score,
            'plot_title': "GRAVY Score Trajectories",
            'y_label': "GRAVY Score",
            'use_random_neurons': False,  # Use matched neurons
            'a': 5.0,
            'b': 3.0
        },
        {
            'pos_match': "high molecular",
            'neg_match': "low molecular",
            'compute_metric_func': compute_molecular_weight,
            'plot_title': "Molecular Weight Trajectories",
            'y_label': "Molecular Weight",
            'use_random_neurons': False,  # Use matched neurons
            'a': 5.0,
            'b': 3.0
        },
        {
            'pos_match': "positive gravy score",
            'neg_match': "negative gravy score",
            'compute_metric_func': compute_gravy_score,
            'plot_title': "Control Experiment (Random Neurons)",
            'y_label': "Instability Index Score",
            'use_random_neurons': True,  # Use random neurons as control
            'a': 5.0,
            'b': 3.0,
            'num_random_neurons': 1
        }
    ]
    
    run_multiple_steering_experiments(
        steering_configs=steering_configs,
        csv_output_path="steered_sequences_8M_multiple.csv"
    )

# Run both steering loops (original functionality preserved)
if __name__ == "__main__":
    # Example usage of the new scalable system
    run_experiments()
