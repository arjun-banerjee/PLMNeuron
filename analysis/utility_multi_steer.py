# # import torch
# # from transformers import AutoModelForMaskedLM, AutoTokenizer
# # from collections import defaultdict
# # import random
# # from tqdm import tqdm
# # import pandas as pd
# # from utility import (
# #     find_matching_neurons, make_multi_neuron_hook, random_protein_sequence, sample_from_logits, decode_tokens,
# #     compute_gravy_score, compute_molecular_weight, compute_instability_index, clean_sequence, compute_charge_at_ph7, CSV_PATH, MODEL_NAME, SEQ_LEN
# # )

# # NUM_STEPS = 100

# # def clean_label(label: str) -> str:
# #     replacements = {
# #         "not positive": "negative",
# #         "not negative": "positive",
# #         "not low": "high",
# #         "not high": "low"
# #     }
# #     # Replace each match (handle compound expressions like "not positive + not high")
# #     for old, new in replacements.items():
# #         label = label.replace(old, new)
# #     return label

# # def multi_goal_steering(
# #     x_match, y_match,
# #     compute_metric_funcs,  # List of functions
# #     metric_names,         # List of metric names (strings)
# #     plot_title="Multi-goal Steering",
# #     y_label="Metric Value",
# #     a=5.0, b=3.0, num_random_neurons=50, csv_output_path="multi_goal_steering.csv"
# # ):
# #     """
# #     Run steering for all 4 combinations of two steering goals (X and Y) by unioning the sets of neurons selected by keyword search for X and Y.
# #     Args:
# #         x_match: Keyword for goal X (e.g., "low molecular weight")
# #         y_match: Keyword for goal Y (e.g., "high GRAVY score")
# #         compute_metric_funcs: List of functions to compute metrics from sequence
# #         metric_names: List of metric names (strings)
# #         plot_title: Title for the plot
# #         y_label: Y-axis label for the plot
# #         a, b: Steering parameters
# #         num_random_neurons: For control, if needed
# #         csv_output_path: Where to save results
# #     """
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
# #     model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
# #     model.eval()

# #     # Generate initial sequence
# #     banned_symbols = {'U', 'B', 'Z', 'X', 'O'}
# #     max_restarts = 10000
# #     base_sequence = random_protein_sequence(SEQ_LEN)
# #     while any(sym in str(base_sequence) for sym in banned_symbols):
# #         if max_restarts == 0:
# #             print("Too many sequence retries")
# #             exit(1)
# #         base_sequence = random_protein_sequence(SEQ_LEN)
# #         max_restarts -= 1
# #     print("Starting sequence:", base_sequence)

# #     not_x_match = clean_label(f"not {x_match}")
# #     not_y_match = clean_label(f"not {y_match}")

# #     # Get neuron sets
# #     x_neurons = set(find_matching_neurons(CSV_PATH, x_match))
# #     y_neurons = set(find_matching_neurons(CSV_PATH, y_match))
# #     not_x_neurons = set(find_matching_neurons(CSV_PATH, not_x_match))
# #     not_y_neurons = set(find_matching_neurons(CSV_PATH, not_y_match))

# #     # take max of k neurons for each set
# #     k = 200
# #     x_neurons = set(list(x_neurons)[:k])
# #     y_neurons = set(list(y_neurons)[:k])
# #     not_x_neurons = set(list(not_x_neurons)[:k])
# #     not_y_neurons = set(list(not_y_neurons)[:k])
    

# #     # 4 combinations: (X & Y), (X & not Y), (not X & Y), (not X & not Y)
# #     combos = [
# #         (x_neurons | y_neurons, f"{x_match} + {y_match}"),
# #         (x_neurons | not_y_neurons, f"{x_match} + {not_y_match}"),
# #         (not_x_neurons | y_neurons, f"{not_x_match} + {y_match}"),
# #         (not_x_neurons | not_y_neurons, f"{not_x_match} + {not_y_match}"),
# #     ]

# #     print("num x_neurons:", len(x_neurons))
# #     print("num y_neurons:", len(y_neurons))
# #     print("num not_x_neurons:", len(not_x_neurons))
# #     print("num not_y_neurons:", len(not_y_neurons))
# #     print("num x and y neurons:", len(combos[0][0]))
# #     print("num x and not y neurons:", len(combos[1][0]))
# #     print("num not x and y neurons:", len(combos[2][0]))
# #     print("num not x and not y neurons:", len(combos[3][0]))

# #     all_results = []
# #     for i, (neuron_set, label) in enumerate(combos):
# #         layer_to_neurons = defaultdict(list)
# #         for layer, neuron in neuron_set:
# #             layer_to_neurons[layer].append(neuron)
# #         seq = base_sequence
# #         history = []
# #         for step in tqdm(range(NUM_STEPS), desc=f"Steering for {label}"):
# #             inputs = tokenizer(seq, return_tensors="pt").to(device)
# #             handles = []
# #             for layer, neurons in layer_to_neurons.items():
# #                 hook = model.base_model.encoder.layer[layer].intermediate.register_forward_hook(
# #                     make_multi_neuron_hook(neurons, a, b)
# #                 )
# #                 handles.append(hook)
# #             with torch.no_grad():
# #                 outputs = model(**inputs, output_hidden_states=True)
# #                 logits = outputs.logits[0]
# #                 hidden_states = outputs.hidden_states
# #             for h in handles:
# #                 h.remove()
# #             # Score: average absolute activation across target neurons
# #             total, count = 0, 0
# #             for layer, neurons in layer_to_neurons.items():
# #                 h = hidden_states[layer][0]
# #                 for n in neurons:
# #                     total += abs(h[:, n].mean().item())
# #                     count += 1
# #             avg_act = total / count if count > 0 else 0
# #             sampled_ids = sample_from_logits(logits)
# #             seq = decode_tokens(tokenizer, sampled_ids)
# #             metric_values = [func(seq) for func in compute_metric_funcs]
# #             history.append((step + 1, seq, avg_act, *metric_values, label))
# #         # Add initial point
# #         initial_metric_values = [func(base_sequence) for func in compute_metric_funcs]
# #         init_row = (0, base_sequence, 0.0, *initial_metric_values, label)
# #         history = [init_row] + history
# #         for row in history:
# #             # row: (step, seq, act, metric1, metric2, ..., label)
# #             step = row[0]
# #             seq = row[1]
# #             act = row[2]
# #             metrics = row[3:-1]
# #             label_val = row[-1]
# #             all_results.append([i, step, seq, act, *metrics, label_val, x_match, y_match])
# #     # Save results
# #     columns = [
# #         "experiment_id", "step", "sequence", "activation"
# #     ] + metric_names + ["label", "x_match_string", "y_match_string"]
# #     df = pd.DataFrame(all_results, columns=columns)
# #     df.to_csv(csv_output_path, index=False)
# #     print(f"\nAll results saved to: {csv_output_path}")
# #     return df

# # # Example usage:
# # if __name__ == "__main__":
# #     df = multi_goal_steering(
# #         x_match="low instability index",
# #         y_match="positive gravy",
# #         compute_metric_funcs=[compute_instability_index, compute_gravy_score],
# #         metric_names=["instability_index", "positive_gravy"],
# #         plot_title="Multi-goal Steering",
# #         y_label="Instability Index & GRAVY Score",
# #         a=1.3, 
# #         b=3,
# #     ) 


# import torch
# from transformers import AutoModelForMaskedLM, AutoTokenizer
# from collections import defaultdict
# import random
# from tqdm import tqdm
# import pandas as pd
# from utility import (
#     find_matching_neurons, make_multi_neuron_hook, random_protein_sequence, sample_from_logits, decode_tokens,
#     compute_gravy_score, compute_molecular_weight, compute_instability_index, clean_sequence, CSV_PATH, MODEL_NAME, SEQ_LEN
# )

# NUM_STEPS = 100

# # New steer function as per user-provided logic

# def steer(model, tokenizer, base_sequence, match_string, label=None, compute_metric_funcs=None, metric_names=None, A=1.3, B=3.0):
#     matched_neurons = find_matching_neurons(CSV_PATH, match_string)
#     if not matched_neurons:
#         print(f"No matching neurons found for: {match_string}")
#         return []

#     layer_to_neurons = defaultdict(list)
#     for layer, neuron in matched_neurons:
#         layer_to_neurons[layer].append(neuron)

#     device = model.device
#     history = []
#     seq = base_sequence
#     for step in tqdm(range(NUM_STEPS), desc=f"Steering for {label}"):
#         masked_seq = list(seq)
#         mask_indices = random.sample(range(len(seq)), len(seq)//4)  # Mask ~25% of positions
#         for idx in mask_indices:
#             masked_seq[idx] = tokenizer.mask_token
#         masked_seq_str = ''.join(masked_seq)
#         inputs = tokenizer(masked_seq_str, return_tensors="pt").to(device)
#         handles = []
#         for layer, neurons in layer_to_neurons.items():
#             hook = model.base_model.encoder.layer[layer].intermediate.register_forward_hook(
#                 make_multi_neuron_hook(neurons, A, B)
#             )
#             handles.append(hook)
#         with torch.no_grad():
#             outputs = model(**inputs, output_hidden_states=True)
#             logits = outputs.logits[0]
#             hidden_states = outputs.hidden_states
#         for h in handles:
#             h.remove()
#         new_seq = list(seq)
#         for idx in mask_indices:
#             token_logits = logits[idx + 1]  # +1 for [CLS] token offset
#             sampled_token_id = sample_from_logits(token_logits.unsqueeze(0))
#             sampled_token_id = sampled_token_id.item()
#             new_token = tokenizer.decode([sampled_token_id], skip_special_tokens=True)
#             if new_token and new_token in "ACDEFGHIKLMNPQRSTVWY":
#                 new_seq[idx] = new_token
#         seq = ''.join(new_seq)
#         total, count = 0, 0
#         for layer, neurons in layer_to_neurons.items():
#             h = hidden_states[layer][0]
#             for n in neurons:
#                 total += abs(h[:, n].mean().item())
#                 count += 1
#         avg_act = total / count if count > 0 else 0
#         # Compute all metrics
#         metric_values = [func(seq) for func in compute_metric_funcs] if compute_metric_funcs else []
#         history.append((step + 1, seq, avg_act, *metric_values, label))
#     return history

# def clean_label(label: str) -> str:
#     replacements = {
#         "not positive": "negative",
#         "not negative": "positive",
#         "not low": "high",
#         "not high": "low"
#     }
#     for old, new in replacements.items():
#         label = label.replace(old, new)
#     return label

# def multi_goal_steering(
#     x_match, y_match,
#     compute_metric_funcs,  # List of functions
#     metric_names,         # List of metric names (strings)
#     plot_title="Multi-goal Steering",
#     y_label="Metric Value",
#     a=5.0, b=3.0, num_random_neurons=50, csv_output_path="multi_goal_steering.csv"
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
#     model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
#     model.eval()
#     banned_symbols = {'U', 'B', 'Z', 'X', 'O'}
#     max_restarts = 10000
#     base_sequence = random_protein_sequence(SEQ_LEN)
#     while any(sym in str(base_sequence) for sym in banned_symbols):
#         if max_restarts == 0:
#             print("Too many sequence retries")
#             exit(1)
#         base_sequence = random_protein_sequence(SEQ_LEN)
#         max_restarts -= 1
#     print("Starting sequence:", base_sequence)
#     not_x_match = clean_label(f"not {x_match}")
#     not_y_match = clean_label(f"not {y_match}")
#     combos = [
#         (x_match, f"{x_match} + {y_match}"),
#         (f"{x_match} + {not_y_match}", f"{x_match} + {not_y_match}"),
#         (f"{not_x_match} + {y_match}", f"{not_x_match} + {y_match}"),
#         (f"{not_x_match} + {not_y_match}", f"{not_x_match} + {not_y_match}"),
#     ]
#     # Actually, for each combo, union the neurons as before
#     x_neurons = set(find_matching_neurons(CSV_PATH, x_match))
#     y_neurons = set(find_matching_neurons(CSV_PATH, y_match))
#     not_x_neurons = set(find_matching_neurons(CSV_PATH, not_x_match))
#     not_y_neurons = set(find_matching_neurons(CSV_PATH, not_y_match))
#     k = 200
#     x_neurons = set(list(x_neurons)[:k])
#     y_neurons = set(list(y_neurons)[:k])
#     not_x_neurons = set(list(not_x_neurons)[:k])
#     not_y_neurons = set(list(not_y_neurons)[:k])
#     neuron_sets = [
#         (x_neurons | y_neurons, f"{x_match} + {y_match}"),
#         (x_neurons | not_y_neurons, f"{x_match} + {not_y_match}"),
#         (not_x_neurons | y_neurons, f"{not_x_match} + {y_match}"),
#         (not_x_neurons | not_y_neurons, f"{not_x_match} + {not_y_match}"),
#     ]
#     all_results = []
#     for i, (neuron_set, label) in enumerate(neuron_sets):
#         # For each combo, create a temporary match string for steer()
#         # We'll pass the unioned neuron set directly
#         # So, we need a custom steer function for this
#         def steer_with_neuron_set(model, tokenizer, base_sequence, neuron_set, label=None, compute_metric_funcs=None, metric_names=None, A=a, B=b):
#             layer_to_neurons = defaultdict(list)
#             for layer, neuron in neuron_set:
#                 layer_to_neurons[layer].append(neuron)
#             device = model.device
#             history = []
#             seq = base_sequence
#             for step in tqdm(range(NUM_STEPS), desc=f"Steering for {label}"):
#                 masked_seq = list(seq)
#                 mask_indices = random.sample(range(len(seq)), len(seq)//4)
#                 for idx in mask_indices:
#                     masked_seq[idx] = tokenizer.mask_token
#                 masked_seq_str = ''.join(masked_seq)
#                 inputs = tokenizer(masked_seq_str, return_tensors="pt").to(device)
#                 handles = []
#                 for layer, neurons in layer_to_neurons.items():
#                     hook = model.base_model.encoder.layer[layer].intermediate.register_forward_hook(
#                         make_multi_neuron_hook(neurons, A, B)
#                     )
#                     handles.append(hook)
#                 with torch.no_grad():
#                     outputs = model(**inputs, output_hidden_states=True)
#                     logits = outputs.logits[0]
#                     hidden_states = outputs.hidden_states
#                 for h in handles:
#                     h.remove()
#                 new_seq = list(seq)
#                 for idx in mask_indices:
#                     token_logits = logits[idx + 1]
#                     sampled_token_id = sample_from_logits(token_logits.unsqueeze(0))
#                     sampled_token_id = sampled_token_id.item()
#                     new_token = tokenizer.decode([sampled_token_id], skip_special_tokens=True)
#                     if new_token and new_token in "ACDEFGHIKLMNPQRSTVWY":
#                         new_seq[idx] = new_token
#                 seq = ''.join(new_seq)
#                 total, count = 0, 0
#                 for layer, neurons in layer_to_neurons.items():
#                     h = hidden_states[layer][0]
#                     for n in neurons:
#                         total += abs(h[:, n].mean().item())
#                         count += 1
#                 avg_act = total / count if count > 0 else 0
#                 metric_values = [func(seq) for func in compute_metric_funcs] if compute_metric_funcs else []
#                 history.append((step + 1, seq, avg_act, *metric_values, label))
#             return history
#         # Add initial point
#         initial_metric_values = [func(base_sequence) for func in compute_metric_funcs]
#         init_row = (0, base_sequence, 0.0, *initial_metric_values, label)
#         history = [init_row] + steer_with_neuron_set(model, tokenizer, base_sequence, neuron_set, label, compute_metric_funcs, metric_names, a, b)
#         for row in history:
#             step = row[0]
#             seq = row[1]
#             act = row[2]
#             metrics = row[3:-1]
#             label_val = row[-1]
#             all_results.append([i, step, seq, act, *metrics, label_val, x_match, y_match])
#     columns = [
#         "experiment_id", "step", "sequence", "activation"
#     ] + metric_names + ["label", "x_match_string", "y_match_string"]
#     df = pd.DataFrame(all_results, columns=columns)
#     df.to_csv(csv_output_path, index=False)
#     print(f"\nAll results saved to: {csv_output_path}")
#     return df

# # Example usage:
# if __name__ == "__main__":
#     df = multi_goal_steering(
#         x_match="low instability index",
#         y_match="positive gravy",
#         compute_metric_funcs=[compute_instability_index, compute_gravy_score],
#         metric_names=["instability_index", "positive_gravy"],
#         plot_title="Multi-goal Steering",
#         y_label="Instability Index & GRAVY Score",
#         a=1.3, 
#         b=3,
#     ) 

import torch.nn.functional as F
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import defaultdict
import random
from tqdm import tqdm
import pandas as pd
from utility import (
    find_matching_neurons, make_multi_neuron_hook, random_protein_sequence, sample_from_logits, decode_tokens,
    compute_gravy_score, compute_molecular_weight, compute_instability_index, clean_sequence, CSV_PATH, MODEL_NAME, SEQ_LEN
)

NUM_STEPS = 100

def mask_random_tokens(seq, tokenizer, fraction=0.1):
    seq = list(seq)
    n_mask = max(1, int(len(seq) * fraction))
    mask_positions = random.sample(range(len(seq)), n_mask)
    masked_seq = seq.copy()
    for idx in mask_positions:
        masked_seq[idx] = tokenizer.mask_token
    return ''.join(masked_seq), mask_positions


def random_neurons(model, num_random_neurons):
    matched_neurons = []
    num_layers = len(model.base_model.encoder.layer)
    hidden_size = model.config.hidden_size
    for _ in range(num_random_neurons):
        layer = random.randint(0, num_layers - 1)
        neuron = random.randint(0, hidden_size - 1)
        matched_neurons.append((layer, neuron))
    return matched_neurons


def steer_mlm(model, tokenizer, base_seq, match_string, metric, label, a, b, use_random_neurons=False, num_random_neurons=50):
    if use_random_neurons:
        matched_neurons = random_neurons(model, num_random_neurons)
    else:
        matched_neurons = find_matching_neurons(CSV_PATH, match_string)
    if not matched_neurons:
        print(f"No matching neurons found for: {match_string}")
        return []
    print(matched_neurons)

    layer_to_neurons = defaultdict(list)
    for layer, neuron in matched_neurons:
        layer_to_neurons[layer].append(neuron)

    device = model.device
    history = []

    seq = base_seq
    for step in tqdm(range(NUM_STEPS), desc=f"Steering: {label}"):
        masked_seq, mask_positions = mask_random_tokens(seq, tokenizer, fraction=0.1)
        inputs = tokenizer(masked_seq, return_tensors="pt").to(device)

        handles = []
        for layer, neurons in layer_to_neurons.items():
            hook = model.base_model.encoder.layer[layer].intermediate.register_forward_hook(
                make_multi_neuron_hook(neurons, a, b)
            )
            handles.append(hook)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0]

        for h in handles:
            h.remove()

        input_ids = inputs["input_ids"][0]
        mask_token_id = tokenizer.mask_token_id
        mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        token_ids = inputs["input_ids"][0].tolist()
        decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        offset = 1 if decoded_tokens[0] in tokenizer.all_special_tokens else 0

        for idx in mask_indices:
            pos = idx.item()
            probs = F.softmax(logits[pos], dim=-1)
            sampled_id = torch.multinomial(probs, num_samples=1).item()
            predicted_token = tokenizer.convert_ids_to_tokens(sampled_id)
            if predicted_token in set("ACDEFGHIKLMNPQRSTVWY"):
                token_ids[pos] = sampled_id
            else:
                print(f"[Warning] Skipped invalid token: {predicted_token}")

        seq = tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")
        log_metric = metric(seq)

        hidden_states = outputs.hidden_states
        total, count = 0, 0
        for layer, neurons in layer_to_neurons.items():
            h = hidden_states[layer + 1][0]  # +1 for embedding offset
            for n in neurons:
                total += abs(h[:, n].mean().item())
                count += 1
        avg_act = total / count if count > 0 else 0

        history.append((step + 1, seq, avg_act, log_metric, label))

    return history

def clean_label(label: str) -> str:
    replacements = {
        "not positive": "negative",
        "not negative": "positive",
        "not low": "high",
        "not high": "low"
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label

def multi_goal_steering(
    x_match, y_match,
    compute_metric_funcs,  # List of functions
    metric_names,         # List of metric names (strings)
    plot_title="Multi-goal Steering",
    y_label="Metric Value",
    a=5.0, b=3.0, num_random_neurons=50, csv_output_path="multi_goal_steering.csv"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
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
    not_x_match = clean_label(f"not {x_match}")
    not_y_match = clean_label(f"not {y_match}")

    x_neurons = set(find_matching_neurons(CSV_PATH, x_match))
    y_neurons = set(find_matching_neurons(CSV_PATH, y_match))
    not_x_neurons = set(find_matching_neurons(CSV_PATH, not_x_match))
    not_y_neurons = set(find_matching_neurons(CSV_PATH, not_y_match))

    # new_labels = {
    #         0: ("High Instability", "Low Instability"),
    #         1: ("Positive GRAVY", "Negative GRAVY"),
    #         2: ("High Mol. Weight", "Low Mol. Weight"),
    #         3: ("Random Neurons A", "Random Neurons B"),
    #     }

    # # map x_match, y_match, not_x_match, not_y_match to new labels
    # x_match = new_labels.get(x_match, (x_match, x_match))[0]
    # y_match = new_labels.get(y_match, (y_match, y_match))[0]
    # not_x_match = new_labels.get(not_x_match, (not_x_match, not_x
    # k = 200
    # x_neurons = set(list(x_neurons)[:k])
    # y_neurons = set(list(y_neurons)[:k])
    # not_x_neurons = set(list(not_x_neurons)[:k])
    # not_y_neurons = set(list(not_y_neurons)[:k])
    neuron_sets = [
        (x_neurons | y_neurons, f"{x_match} + {y_match}"),
        (x_neurons | not_y_neurons, f"{x_match} + {not_y_match}"),
        (not_x_neurons | y_neurons, f"{not_x_match} + {y_match}"),
        (not_x_neurons | not_y_neurons, f"{not_x_match} + {not_y_match}"),
    ]

    # neuron_sets = [
    #     (x_neurons ^ y_neurons, f"{x_match} + {y_match}"),
    #     (x_neurons ^ not_y_neurons, f"{x_match} + {not_y_match}"),
    #     (not_x_neurons ^ y_neurons, f"{not_x_match} + {y_match}"),
    #     (not_x_neurons ^ not_y_neurons, f"{not_x_match} + {not_y_match}"),
    # ]
    all_results = []
    for i, (neuron_set, label) in enumerate(neuron_sets):
        # For each combo, run steer_mlm with the unioned neuron set
        def metric(seq):
            return tuple(func(seq) for func in compute_metric_funcs)
        # Patch steer_mlm to accept neuron_set directly
        def steer_mlm_with_neuron_set(model, tokenizer, base_seq, neuron_set, metric, label, a, b):
            layer_to_neurons = defaultdict(list)
            for layer, neuron in neuron_set:
                layer_to_neurons[layer].append(neuron)
            device = model.device
            history = []
            seq = base_seq
            for step in tqdm(range(NUM_STEPS), desc=f"Steering: {label}"):
                masked_seq, mask_positions = mask_random_tokens(seq, tokenizer, fraction=0.1)
                inputs = tokenizer(masked_seq, return_tensors="pt").to(device)
                handles = []
                for layer, neurons in layer_to_neurons.items():
                    hook = model.base_model.encoder.layer[layer].intermediate.register_forward_hook(
                        make_multi_neuron_hook(neurons, a, b)
                    )
                    handles.append(hook)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    logits = outputs.logits[0]
                for h in handles:
                    h.remove()
                input_ids = inputs["input_ids"][0]
                mask_token_id = tokenizer.mask_token_id
                mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
                token_ids = inputs["input_ids"][0].tolist()
                decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
                offset = 1 if decoded_tokens[0] in tokenizer.all_special_tokens else 0
                for idx in mask_indices:
                    pos = idx.item()
                    probs = F.softmax(logits[pos], dim=-1)
                    sampled_id = torch.multinomial(probs, num_samples=1).item()
                    predicted_token = tokenizer.convert_ids_to_tokens(sampled_id)
                    if predicted_token in set("ACDEFGHIKLMNPQRSTVWY"):
                        token_ids[pos] = sampled_id
                    else:
                        print(f"[Warning] Skipped invalid token: {predicted_token}")
                seq = tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")
                log_metric = metric(seq)
                history.append((step + 1, seq, 0.0, *log_metric, label))
            return history
        initial_metric_values = [func(base_sequence) for func in compute_metric_funcs]
        init_row = (0, base_sequence, 0.0, *initial_metric_values, label)
        history = [init_row] + steer_mlm_with_neuron_set(model, tokenizer, base_sequence, neuron_set, metric, label, a, b)
        for row in history:
            step = row[0]
            seq = row[1]
            act = row[2]
            metrics = row[3:-1]
            label_val = row[-1]
            all_results.append([i, step, seq, act, *metrics, label_val, x_match, y_match])
    columns = [
        "experiment_id", "step", "sequence", "activation"
    ] + metric_names + ["label", "x_match_string", "y_match_string"]
    df = pd.DataFrame(all_results, columns=columns)
    df.to_csv(csv_output_path, index=False)
    print(f"\nAll results saved to: {csv_output_path}")
    return df

# Example usage:
if __name__ == "__main__":
    torch.manual_seed(33)
    random.seed(33)
    df = multi_goal_steering(
        x_match="high molecular",
        y_match="high gravy",
        compute_metric_funcs=[compute_molecular_weight, compute_gravy_score],
        metric_names=["molecular_weight", "positive_gravy"],
        plot_title="Multi-goal Steering",
        y_label="Molecular Weight & GRAVY Score",
        a=10, 
        b=3,
    ) 