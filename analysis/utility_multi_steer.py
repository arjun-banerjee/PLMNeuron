import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import defaultdict
import random
from tqdm import tqdm
import pandas as pd
from utility import (
    find_matching_neurons, make_multi_neuron_hook, random_protein_sequence, sample_from_logits, decode_tokens,
    compute_gravy_score, compute_molecular_weight, compute_instability_index, clean_sequence, compute_charge_at_ph7, CSV_PATH, MODEL_NAME, SEQ_LEN, NUM_STEPS
)

def multi_goal_steering(
    x_match, y_match,
    compute_metric_funcs,  # List of functions
    metric_names,         # List of metric names (strings)
    plot_title="Multi-goal Steering",
    y_label="Metric Value",
    a=5.0, b=3.0, num_random_neurons=50, csv_output_path="multi_goal_steering.csv"
):
    """
    Run steering for all 4 combinations of two steering goals (X and Y) by unioning the sets of neurons selected by keyword search for X and Y.
    Args:
        x_match: Keyword for goal X (e.g., "low molecular weight")
        y_match: Keyword for goal Y (e.g., "high GRAVY score")
        compute_metric_funcs: List of functions to compute metrics from sequence
        metric_names: List of metric names (strings)
        plot_title: Title for the plot
        y_label: Y-axis label for the plot
        a, b: Steering parameters
        num_random_neurons: For control, if needed
        csv_output_path: Where to save results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Generate initial sequence
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

    # Get neuron sets
    x_neurons = set(find_matching_neurons(CSV_PATH, x_match))
    y_neurons = set(find_matching_neurons(CSV_PATH, y_match))
    not_x_neurons = set(find_matching_neurons(CSV_PATH, f"not {x_match}"))
    not_y_neurons = set(find_matching_neurons(CSV_PATH, f"not {y_match}"))

    # 4 combinations: (X & Y), (X & not Y), (not X & Y), (not X & not Y)
    combos = [
        (x_neurons | y_neurons, f"{x_match} + {y_match}"),
        (x_neurons | not_y_neurons, f"{x_match} + not {y_match}"),
        (not_x_neurons | y_neurons, f"not {x_match} + {y_match}"),
        (not_x_neurons | not_y_neurons, f"not {x_match} + not {y_match}")
    ]

    all_results = []
    for i, (neuron_set, label) in enumerate(combos):
        layer_to_neurons = defaultdict(list)
        for layer, neuron in neuron_set:
            layer_to_neurons[layer].append(neuron)
        seq = base_sequence
        history = []
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
            metric_values = [func(seq) for func in compute_metric_funcs]
            history.append((step + 1, seq, avg_act, *metric_values, label))
        # Add initial point
        initial_metric_values = [func(base_sequence) for func in compute_metric_funcs]
        init_row = (0, base_sequence, 0.0, *initial_metric_values, label)
        history = [init_row] + history
        for row in history:
            # row: (step, seq, act, metric1, metric2, ..., label)
            step = row[0]
            seq = row[1]
            act = row[2]
            metrics = row[3:-1]
            label_val = row[-1]
            all_results.append([i, step, seq, act, *metrics, label_val, x_match, y_match])
    # Save results
    columns = [
        "experiment_id", "step", "sequence", "activation"
    ] + metric_names + ["label", "x_match_string", "y_match_string"]
    df = pd.DataFrame(all_results, columns=columns)
    df.to_csv(csv_output_path, index=False)
    print(f"\nAll results saved to: {csv_output_path}")
    return df

# Example usage:
if __name__ == "__main__":
    df = multi_goal_steering(
        x_match="high charge at pH7",
        y_match="high molecular weight",
        compute_metric_funcs=[compute_charge_at_ph7, compute_instability_index],
        metric_names=["charge_ph7", "instability_index"],
        plot_title="Multi-goal Steering",
        y_label="Charge at pH7 & Instability Index",
    ) 