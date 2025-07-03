import os
import torch
import pandas as pd
from esm import pretrained, Alphabet, BatchConverter

# ========== Config ==========
input_fasta = "/Users/davidm/182-final-proj/data/uniref50.fasta"
output_csv = os.path.join(os.path.dirname(input_fasta), "uniprot50_perplexity.csv")
MAX_SEQUENCES = 10 # keep smaller if running on CPU
MODEL_NAME = "esm2_t12_35M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Load Model ==========
model, alphabet = pretrained.load_model_and_alphabet(MODEL_NAME)
model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()

# ========== Perplexity Function ==========
def compute_perplexity(seq: str):
    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        logits = model(tokens)["logits"]
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0, :-1].gather(1, tokens[0, 1:].unsqueeze(1)).squeeze()
    perplexity = torch.exp(-token_log_probs.mean()).item()
    return perplexity

# ========== Read Sequences ==========
from Bio import SeqIO
sequences = []
perplexities = []
ids = []

with open(input_fasta) as handle:
    for i, record in enumerate(SeqIO.parse(handle, "fasta")):
        if i >= MAX_SEQUENCES:
            break
        seq = str(record.seq)
        if "X" in seq:
            continue
        try:
            ppl = compute_perplexity(seq)
            sequences.append(seq)
            perplexities.append(ppl)
            ids.append(record.id)
            print(f"[{i}] {record.id}: Perplexity = {ppl:.2f}")
        except Exception as e:
            print(f"Error on sequence {record.id}: {e}")

# ========== Save Results ==========
df = pd.DataFrame({
    "sequence_id": ids,
    "sequence": sequences,
    "perplexity": perplexities
})
df.to_csv(output_csv, index=False)
print(f"\nSaved perplexity CSV to: {output_csv}")
