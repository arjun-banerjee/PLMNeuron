import pandas as pd
import re
import requests
from tqdm import tqdm

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from b2bTools import SingleSeq
import peptides
import protpy as protpy


def compute_b2b_features(str: protein_seq) -> dict:
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

def compute_peptides_features(str: protein_seq) -> dict:
    """
    Compute features for a given string using proptpy.
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


def compute_biopython_features(str: protein_seq) -> dict:
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


def compute_proptpy_features(str: protein_seq) -> dict:
    """
    Compute protpy features
    """
    #TODO: Compute features based on this repo https://peptides.readthedocs.io/en/stable/api/descriptors.html#peptides.PRINComponents

    features = {}
    return features


