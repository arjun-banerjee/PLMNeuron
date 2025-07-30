import pandas as pd
import re
import requests
from tqdm import tqdm

from Bio.SeqUtils.ProtParam import ProteinAnalysis
# from b2bTools import SingleSeq
import peptides
import protpy as protpy

def add_dataframe_to_features(df, base_key, features):
    """
    Helper to add each column of a single-row DataFrame to features dict with a base key prefix.
    Example: add_dataframe_to_features(df, 'AminoAcidComp', features) will add keys like 'AminoAcidComp_A', ...
    """
    for col in df.columns:
        features[f"{base_key}_{col}"] = float(df[col].iloc[0])

def compute_b2b_features(protein_seq) -> dict:
    """
    Compute features for a given string using b2bTools.
    """
    #Write the sequence to a fasta file for b2bTools
    fasta = f'>SEQ_1\n{protein_seq}\n'
    with open("input_example1.fasta", "w") as f:
        f.write(fasta)
    
    #Code from https://colab.research.google.com/github/Bio2Byte/public_notebooks/blob/main/Bio2ByteTools_v3_singleseq_pypi.ipynb#scrollTo=oRxE84upgFdn
    #TODO: Do we need to extract features from the b2bTools one by one?
    #TODO: How do we pass context abt what these scores mean into LLM. Would likely need to be in System Prompt? 
    single_seq = SingleSeq("/content/input_example1.fasta")
    single_seq.predict(tools=['dynamine', 'efoldmine', 'disomine', 'agmata', 'psp'])

    all_predictions = single_seq.get_all_predictions()
    return all_predictions

def compute_peptides_features(protein_seq) -> dict:
    """
    Compute peptide features for a given string using the peptides package.
    """
    #All from: https://peptides.readthedocs.io/en/stable/api/peptide.html
    peptide = peptides.Peptide(protein_seq)
    features = {}
    features["Alphatic Index"] = peptide.alphatic_index()
    features["Atchley Factors"] = peptide.atchley_factors()
    features["Blosum Indices"] = peptide.blosum_indices()
    features["Boman Index"] = peptide.boman()
    features["Cruciani Properties"] = peptide.cruciani_properties()
    features["Akashi Energy Cost"] = peptide.energy_cost(scale = "Akashi")
    features["Craig Energy Cost"] = peptide.energy_cost(scale = "Craig")
    features["Heizer Energy Cost"] = peptide.energy_cost(scale = "Heizer")
    features["Wagner Energy Cost"] = peptide.energy_cost(scale = "Wagner")
    features["Fagasi Vectors"] = peptide.fagasi_vectors()
    features["Hydrophobic Moment"] = peptide.hydrophobic_moment()
    features["MS Whim Score"] = peptide.ms_whim_score()
    features["Mass over Charge Ration"] = peptide.mz()
    features["Nutrient Cost"] = peptide.nutrient_cost()
    features["Physical Chemical Properties"] = peptide.pcp_descriptors()
    features["PRIN Components"] = peptide.prin_components()
    features["Protfp Descriptors"] = peptide.protfp_descriptors()
    features["Sneath Vectors"] = peptide.sneath_vectors()
    features["ST Scales"] = peptide.st_scales()
    features["Structural Class"] = peptide.structural_class()
    features["SVGR"] = peptide.svger_descriptors()
    features["T Scales"] = peptide.t_scales()
    features["VHSE Scales"] = peptide.vhse_scales()
    features["VSTPV Descriptors"] = peptide.vstpv_descriptors()
    features["Z Scales"] = peptides.z_scales()

    return features


def compute_biopython_features(protein_seq) -> dict:
    """
    Compute manual features for a given string.
    """
    features = {}

    #BioPython ProteinAnalysis
    #https://biopython.org/docs/1.76/api/Bio.SeqUtils.ProtParam.html
    protein_analysis = ProteinAnalysis(protein_seq)
    features["Amino Acid Percentages"] = protein_analysis.get_amino_acids_percent()
    features["Molecular Weight"] = protein_analysis.molecular_weight()
    features["Aromaticity"] = protein_analysis.aromaticity()
    features["Instability Index"] = protein_analysis.instability_index()
    #TODO: Figure out flexibility
    features["GRAVY"] = protein_analysis.gravy()
    #TODO: Figure out protein scale 
    features["Isoelectric Point"] = protein_analysis.isoelectric_point()
    #TODO: Figure out what pH's are relevant 
    features["Charge at pH 7"] = protein_analysis.charge_at_pH(7.4)
    secondary_structure = protein_analysis.secondary_structure_fraction()
    features["Helix Fraction"] = secondary_structure[0]
    features["Turn Fraction"] = secondary_structure[1]
    features["Sheet Fraction"] = secondary_structure[2]
    features["Molecular Extinction Coefficient"] = protein_analysis.molar_extinction_coefficient()

    return features


def compute_protpy_features(protein_seq) -> dict:
    """
    Compute features for a given string using the protpy package.
    """
    amino_acid_composition = protpy.amino_acid_composition(protein_seq)
    dipeptide_composition = protpy.dipeptide_composition(protein_seq)
    tripeptide_composition = protpy.tripeptide_composition(protein_seq)
    moreaubroto_autocorrelation = protpy.moreaubroto_autocorrelation(protein_seq) 
    moran_autocorrelation = protpy.moran_autocorrelation(protein_seq)
    geary_autocorrelation = protpy.geary_autocorrelation(protein_seq)
    conjoint_triad = protpy.conjoint_triad(protein_seq)
    ctd_composition = protpy.ctd_composition(protein_seq)
    socn_all = protpy.sequence_order_coupling_number(protein_seq)
    qso = protpy.quasi_sequence_order(protein_seq)

    features = {}
    add_dataframe_to_features(amino_acid_composition, "AminoAcidComp", features)
    add_dataframe_to_features(dipeptide_composition, "DipeptideComp", features)
    add_dataframe_to_features(tripeptide_composition, "TripeptideComp", features)
    add_dataframe_to_features(moreaubroto_autocorrelation, "MoreauBrotoAutoCorr", features)
    add_dataframe_to_features(moran_autocorrelation, "MoranAutoCorr", features)
    add_dataframe_to_features(geary_autocorrelation, "GearyAutoCorr", features)
    add_dataframe_to_features(conjoint_triad, "ConjointTriad", features)
    add_dataframe_to_features(ctd_composition, "CTDComp", features)
    add_dataframe_to_features(socn_all, "SOCN", features)
    add_dataframe_to_features(qso, "QSO", features)
    
    return features


if __name__ == "__main__":
    protein_seq = "LYLIFGAWAGMVGTALSLLIRAEL"
    features = compute_protpy_features(protein_seq)
    print("started")
    for k, v in list(features.items())[:20]:  # print only first 20 for brevity
        print(f"{k}: {v}")
    print(f"Total features: {len(features)}")


