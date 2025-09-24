import pandas as pd
import re
import requests
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import peptides
import protpy as protpy
import argparse
import multiprocessing
import csv
import os
from multiprocessing import Lock as MPILock

# Global variable to hold the lock within each worker process
csv_lock = None

counter = multiprocessing.Value('i', 0)


def init_worker(lock):
    global csv_lock
    csv_lock = lock

def add_dataframe_to_features(df, base_key, features):
    for col in df.columns:
        features[f"{base_key}_{col}"] = float(df[col].iloc[0])

def compute_peptides_features(protein_seq) -> dict:
    peptide = peptides.Peptide(protein_seq)
    features = {}
    try:
        features["Aliphatic Index"] = peptide.aliphatic_index()
        features["Atchley Factors"] = peptide.atchley_factors()
        features["Blosum Indices"] = peptide.blosum_indices()
        features["Boman Index"] = peptide.boman()
        features["Cruciani Properties"] = peptide.cruciani_properties()
        features["Akashi Energy Cost"] = peptide.energy_cost(scale="Akashi")
        features["Craig Energy Cost"] = peptide.energy_cost(scale="Craig")
        features["Heizer Energy Cost"] = peptide.energy_cost(scale="Heizer")
        features["Wagner Energy Cost"] = peptide.energy_cost(scale="Wagner")
        features["Hydrophobic Moment"] = peptide.hydrophobic_moment()
        features["Mass over Charge Ratio"] = peptide.mz()
        features["Nutrient Cost"] = peptide.nutrient_cost()
        features["Physical Chemical Properties"] = peptide.pcp_descriptors()
        features["PRIN Components"] = peptide.prin_components()
        features["Protfp Descriptors"] = peptide.protfp_descriptors()
        features["Sneath Vectors"] = peptide.sneath_vectors()
        features["ST Scales"] = peptide.st_scales()
        features["Structural Class"] = peptide.structural_class()
    except Exception as e:
        print(f"[Peptides fail] {protein_seq[:10]}...: {e}")
    return features

def compute_biopython_features(protein_seq) -> dict:
    features = {}
    try:
        protein_analysis = ProteinAnalysis(protein_seq)
        features["Amino Acid Percentages"] = protein_analysis.get_amino_acids_percent()
        features["Molecular Weight"] = protein_analysis.molecular_weight()
        features["Aromaticity"] = protein_analysis.aromaticity()
        features["Instability Index"] = protein_analysis.instability_index()
        features["GRAVY"] = protein_analysis.gravy()
        features["Isoelectric Point"] = protein_analysis.isoelectric_point()
        features["Charge at pH 7"] = protein_analysis.charge_at_pH(7.4)
        secondary_structure = protein_analysis.secondary_structure_fraction()
        features["Helix Fraction"] = secondary_structure[0]
        features["Turn Fraction"] = secondary_structure[1]
        features["Sheet Fraction"] = secondary_structure[2]
        features["Molecular Extinction Coefficient"] = protein_analysis.molar_extinction_coefficient()
    except Exception as e:
        print(f"[BioPython fail] {protein_seq[:10]}...: {e}")
    return features

def compute_protpy_features(protein_seq) -> dict:
    features = {}
    try:
        #amino_acid_composition = protpy.amino_acid_composition(protein_seq)
        dipeptide_composition = protpy.dipeptide_composition(protein_seq)
        tripeptide_composition = protpy.tripeptide_composition(protein_seq)
        moreaubroto_autocorrelation = protpy.moreaubroto_autocorrelation(protein_seq)
        moran_autocorrelation = protpy.moran_autocorrelation(protein_seq)
        geary_autocorrelation = protpy.geary_autocorrelation(protein_seq)
        conjoint_triad = protpy.conjoint_triad(protein_seq)
        ctd_composition = protpy.ctd_composition(protein_seq)
        socn_all = protpy.sequence_order_coupling_number(protein_seq)
        qso = protpy.quasi_sequence_order(protein_seq)

        #add_dataframe_to_features(amino_acid_composition, "AminoAcidComp", features)
        add_dataframe_to_features(dipeptide_composition, "DipeptideComp", features)
        add_dataframe_to_features(tripeptide_composition, "TripeptideComp", features)
        add_dataframe_to_features(moreaubroto_autocorrelation, "MoreauBrotoAutoCorr", features)
        add_dataframe_to_features(moran_autocorrelation, "MoranAutoCorr", features)
        add_dataframe_to_features(geary_autocorrelation, "GearyAutoCorr", features)
        add_dataframe_to_features(conjoint_triad, "ConjointTriad", features)
        add_dataframe_to_features(ctd_composition, "CTDComp", features)
        add_dataframe_to_features(socn_all, "SOCN", features)
        add_dataframe_to_features(qso, "QSO", features)
    except Exception as e:
        print(f"[ProtPy fail] {protein_seq[:10]}...: {e}")
    return features

def load_existing_sequences(output_file):
    """Return set of sequences already processed in the output CSV."""
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return set()
    try:
        existing = pd.read_csv(output_file, usecols=["Sequence"])
        return set(existing["Sequence"].dropna().astype(str))
    except Exception as e:
        print(f"[Main] Could not load existing sequences: {e}")
        return set()


def process_batch(batch, output_file, existing_sequences):
    results_to_write = []
    
    for seq, row_data in batch:
        if seq in existing_sequences:   # <-- Skip if already written
            continue

        features = {}
        try:
            features.update(compute_protpy_features(seq))
        except Exception as e:
            print(f"[ProtPy error] {seq[:10]}...: {e}")
        
        try:
            features.update(compute_peptides_features(seq))
        except Exception as e:
            print(f"[Peptides error] {seq[:10]}...: {e}")

        result_row = {**row_data, **features}
        results_to_write.append(result_row)
    
    if not results_to_write:
        print(f"[Process {os.getpid()}] No new results to write.")
        return
    
    with csv_lock:
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', newline='') as f:
            fieldnames = list(results_to_write[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists or os.path.getsize(output_file) == 0:
                print(f"[Process {os.getpid()}] Writing CSV header.")
                writer.writeheader()
            
            writer.writerows(results_to_write)
    print(f"[Process {os.getpid()}] Finished batch.")


def main():
    parser = argparse.ArgumentParser(description="Process protein sequences to extract features.")
    parser.add_argument("input_file", help="Path to the input TSV.GZ file containing protein data.")
    parser.add_argument("--output", dest="output_file", default="protein_features.csv",
                        help="Path for the output CSV file (default: protein_features.csv)")
    args = parser.parse_args()

    print(f"[Main] Reading data from {args.input_file}...")
    df = pd.read_csv(args.input_file, sep='\t', compression='gzip', low_memory=False)
    print(f"[Main] Loaded {len(df)} rows.")

    # Load already processed sequences
    existing_sequences = load_existing_sequences(args.output_file)
    print(f"[Main] Found {len(existing_sequences)} existing sequences in {args.output_file}")

    data_to_process = [(row['Sequence'], row.to_dict()) for _, row in df.iterrows()]
    
    num_processes = multiprocessing.cpu_count()
    batch_size = min(max(1, len(data_to_process) // num_processes), 256)
    batches = [data_to_process[i:i + batch_size] for i in range(0, len(data_to_process), batch_size)]
    
    print(f"[Main] Starting parallel processing with {num_processes} processes and {len(batches)} batches (batch size {batch_size}).")

    main_lock = MPILock()

    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(main_lock,)) as pool:
        list(tqdm(pool.starmap(process_batch, [(batch, args.output_file, existing_sequences) for batch in batches]),
                  total=len(batches)))

    print(f"[Main] Processing complete. Results saved to {args.output_file}")


if __name__ == '__main__':
    main()
