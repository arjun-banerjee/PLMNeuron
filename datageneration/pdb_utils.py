import requests
import os
import MDAnalysis as mda
from MDAnalysis.analysis import hbonds
import numpy as np
import networkx as nx
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.Polypeptide import is_aa


def fetch_pdb(pdb_id: str, out_dir: str = ".") -> str:
    """
    Download a PDB file given its PDB ID from RCSB and save it locally.

    Args:
        pdb_id (str): The 4-character PDB ID (e.g., '1TUP').
        out_dir (str): Directory to save the downloaded PDB file.

    Returns:
        str: Path to the downloaded PDB file.
    """
    pdb_id = pdb_id.upper()
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(pdb_url)

    if response.status_code != 200:
        raise ValueError(f"Failed to fetch PDB ID {pdb_id}. HTTP {response.status_code}")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f"{pdb_id}.pdb")

    with open(pdb_path, "w") as f:
        f.write(response.text)

    print(f"PDB file saved: {pdb_path}")
    return pdb_path

def compute_contact_graph(coords, threshold=8.0):
    """Build a contact graph where residues within threshold distance are connected."""
    n = len(coords)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                G.add_edge(i, j)
    return G

def compute_secondary_structure_counts(model, pdb_file):
    """Compute secondary structure counts using DSSP."""
    dssp = DSSP(model[0], pdb_file)
    ss_counts = {"H": 0, "B": 0, "E": 0, "G": 0, "I": 0, "T": 0, "S": 0, "-": 0}
    for key in dssp.keys():
        ss = dssp[key][2]
        ss_counts[ss] = ss_counts.get(ss, 0) + 1
    return ss_counts

def extract_geometry_features_mda(pdb_path):
    """Extracts radius of gyration, average B-factor, hydrogen bonds, and compactness from a PDB file using MDAnalysis."""
    u = mda.Universe(pdb_path)
    
    # 1. Radius of Gyration (all atoms)
    rg = u.atoms.radius_of_gyration()
    
    # 2. Average B-factor
    avg_b = np.mean(u.atoms.tempfactors) if hasattr(u.atoms, 'tempfactors') else None
    
    # 3. Hydrogen Bonds
    h = hbonds.HydrogenBondAnalysis(u, selection1='protein', selection2='protein', distance=3.5, angle=150.0)
    h.run()
    hbonds_count = len(h.results.hbonds)
    
    # 4. Compactness: residues per bounding-box volume
    coords = u.atoms.positions
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bounding_box_volume = np.prod(maxs - mins)
    num_residues = len(u.residues)
    compactness = num_residues / bounding_box_volume if bounding_box_volume > 0 else 0
    
    return {
        "Radius of Gyration": rg,
        "Average B factor": avg_b,
        "Hydrogen Bonds": hbonds_count,
        "Compactness": compactness
    }

def extract_all_features(pdb_file, chain_id=None):
    """
    Combines Biopython (secondary structure, contact clustering)
    and MDAnalysis-based descriptors into one feature dictionary.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Extract alpha carbon coordinates (for contact graph only)
    coords = []
    chain = None
    for model in structure:
        for ch in model:
            if chain_id is None or ch.id == chain_id:
                chain = ch
                for residue in ch:
                    if is_aa(residue):
                        if 'CA' in residue:
                            coords.append(residue['CA'].coord)
                break
        if chain:
            break
    coords = np.array(coords)
    
    if len(coords) == 0:
        raise ValueError("No CA atoms found in the specified chain or structure.")
    
    # Contact and secondary structure features
    G = compute_contact_graph(coords)
    clustering_coeff = nx.average_clustering(G)
    ss_counts = compute_secondary_structure_counts(structure[0], pdb_file)
    
    # MDAnalysis features (includes radius of gyration)
    mda_features = extract_geometry_features_mda(pdb_file)
    
    # Combine all features
    combined_features = {
        "Avg Clustering Coefficient": clustering_coeff,
        **ss_counts,
        **mda_features
    }
    return combined_features