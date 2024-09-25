import os
import logging
import json
import multiprocessing
import requests
import urllib.parse

from typing import Dict, Any, List
from rdkit import Chem
from rdkit.Chem import Descriptors

from Bio.SeqUtils import molecular_weight, ProtParam
from Bio.Blast import NCBIWWW, NCBIXML

logger = logging.getLogger(__name__)





def filter_ligand_physicochemical(smiles: str) -> (bool, str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES"

    # Calculate properties
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    # Define acceptable ranges
    weight_range = (150, 500)  # molecular weight in g/mol
    logp_range = (-2, 5)       # LogP value for hydrophobicity
    tpsa_limit = 140           # Topological polar surface area in Å²

    # Filter based on ranges
    if not (weight_range[0] <= mol_weight <= weight_range[1]):
        return False, f"Molecular weight {mol_weight} out of range"
    if not (logp_range[0] <= logp <= logp_range[1]):
        return False, f"LogP {logp} out of range"
    if tpsa > tpsa_limit:
        return False, f"TPSA {tpsa} out of range"

    return True, "Ligand passes physicochemical filters"





def filter_protein_physicochemical(sequence: str) -> (bool, str):
    # Calculate molecular weight
    mol_weight = molecular_weight(sequence, seq_type="protein")

    # Use ProtParam to calculate instability index and isoelectric point
    prot_param = ProtParam.ProteinAnalysis(sequence)
    instability_index = prot_param.instability_index()
    iso_point = prot_param.isoelectric_point()

    # Define acceptable ranges
    weight_limit = 150000       # Max molecular weight in Da
    instability_limit = 40      # Instability index threshold
    iso_point_range = (4.0, 9.0)  # Acceptable range for isoelectric point

    # Filter based on properties
    if mol_weight > weight_limit:
        return False, f"Molecular weight {mol_weight} out of range"
    if instability_index > instability_limit:
        return False, f"Instability index {instability_index} too high"
    if not (iso_point_range[0] <= iso_point <= iso_point_range[1]):
        return False, f"Isoelectric point {iso_point} out of range"

    return True, "Protein passes physicochemical filters"





def check_protein_similarity(sequence: str) -> (bool, str):
    try:
        result_handle = NCBIWWW.qblast("blastp", "nr", sequence)
        blast_record = NCBIXML.read(result_handle)
    except Exception as e:
        logger.error(f"Error during BLAST search: {e}")
        return False, f"BLAST search failed: {e}"

    # Check if there is a significant match (E-value < 0.05)
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            if hsp.expect < 0.05:
                return False, f"Sequence too similar to known sequence: {alignment.hit_def}"

    return True, "Sequence is sufficiently novel"



def check_ligand_novelty(smiles: str) -> (bool, str):
    encoded_smiles = urllib.parse.quote(smiles)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/cids/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        if "IdentifierList" in response.json():
            return False, "SMILES string matches a known compound in PubChem"
        return True, "Ligand is novel"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking ligand novelty: {e}")
        return False, f"Failed to check ligand novelty: {e}"





def admet_screening(smiles: str) -> (bool, str):
    # Note: The pkCSM API endpoint may not be publicly accessible.
    # For robustness, we'll handle potential errors.
    url = "https://biosig.lab.uq.edu.au/pkcsm/prediction"
    headers = {'Content-Type': 'application/json'}
    data = {"smiles": [smiles], "admet": "toxicity"}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        results = response.json()
        # Assuming the API returns a specific structure (modify as per actual API)
        if 'toxicity' in results and results['toxicity'][0]['alert']:
            return False, "Toxicity alert detected"
        return True, "No toxicity alerts"
    except requests.exceptions.RequestException as e:
        logger.error(f"ADMET screening failed: {e}")
        return False, f"ADMET screening failed: {e}"
    except KeyError as e:
        logger.error(f"Unexpected response structure: {e}")
        return False, "ADMET screening failed due to unexpected response"




def swiss_adme_screen(smiles: str) -> (bool, str):
    # Note: SwissADME does not provide a public API for programmatic access.
    # The following is a placeholder to illustrate exception handling.
    try:
        # Placeholder for actual API call or method
        response = requests.get(f"http://www.swissadme.ch/index.php?{urllib.parse.quote(smiles)}")
        response.raise_for_status()
        if "No toxic" in response.text:
            return True, "No toxicity predicted by SwissADME"
        else:
            return False, "SwissADME predicts potential toxicity"
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to screen with SwissADME: {e}")
        return False, f"Failed to screen with SwissADME: {e}"


def calculate_ligand_rating(smiles: str) -> (float, Dict[str, Any]):
    # Initialize scores and messages
    physico_score = 0
    novelty_score = 0
    admet_score = 0
    total_score = 0

    # Step 1: Physicochemical filtering
    pass_physico, physico_message = filter_ligand_physicochemical(smiles)
    if not pass_physico:
        return 0, {
            "smiles": smiles,
            "physicochemical_score": physico_score,
            "novelty_score": novelty_score,
            "admet_score": admet_score,
            "total_score": total_score,
            "comments": {
                "physicochemical_message": physico_message,
                "novelty_message": "",
                "admet_message": ""
            }
        }
    else:
        physico_score = 8

    # Step 2: Novelty check
    pass_novelty, novelty_message = check_ligand_novelty(smiles)
    novelty_score = 9 if pass_novelty else 3

    # Step 3: ADMET Screening
    pass_admet, admet_message = admet_screening(smiles)
    admet_score = 10 if pass_admet else 4

    # Calculate overall rating (weighted sum example)
    total_score = (physico_score * 0.4) + (novelty_score * 0.3) + (admet_score * 0.3)
    return total_score, {
        "smiles": smiles,
        "physicochemical_score": physico_score,
        "novelty_score": novelty_score,
        "admet_score": admet_score,
        "total_score": total_score,
        "comments": {
            "physicochemical_message": physico_message,
            "novelty_message": novelty_message,
            "admet_message": admet_message
        }
    }



def store_ligand_result(result: Dict[str, Any]):
    results = []
    results_file = os.path.join(RESULTS_DIR, "ligand_screening_results.json")
    try:
        with open(results_file, "r") as file:
            results = json.load(file)
    except FileNotFoundError:
        pass  # No results file yet, start with an empty list
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file: {e}")
        return

    results.append(result)
    with open(results_file, "w") as file:
        json.dump(results, file, indent=4)





def process_ligand(smiles_file: str) -> Dict[str, Any]:
    with open(os.path.join(SMILES_DIR, smiles_file), 'r') as f:
        smiles = f.read().strip()

    total_score, details = calculate_ligand_rating(smiles)

    ligand_name = os.path.splitext(smiles_file)[0]
    result = {
        "ligand_name": ligand_name,
        "smiles": smiles,
        "total_score": total_score,
        "details": details
    }

    # Store the result
    with open(os.path.join(RESULTS_DIR, f'{ligand_name}_result.json'), 'w') as f:
        json.dump(result, f, indent=4)

    return result









def high_throughput_screening():
    logger.info("Starting high-throughput screening pipeline...")

    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Get all SMILES files
    smiles_files = [f for f in os.listdir(SMILES_DIR) if f.endswith('.smi')]

    # Use multiprocessing to screen ligands in parallel
    with multiprocessing.Pool() as pool:
        results = pool.map(process_ligand, smiles_files)

    # Store all results in a single JSON file
    all_results_file = os.path.join(RESULTS_DIR, 'all_screening_results.json')
    try:
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        logger.error(f"Error writing all results to JSON file: {e}")

    logger.info(f"High-throughput screening completed. Processed {len(smiles_files)} ligands.")
    logger.info(f"Results stored in {RESULTS_DIR}")




def get_top_ligand_results() -> List[Dict[str, Any]]:
    try:
        with open(os.path.join(RESULTS_DIR, 'all_screening_results.json'), 'r') as f:
            all_results = json.load(f)
        # Sort the results by total_score
        sorted_results = sorted(all_results, key=lambda x: x['total_score'], reverse=True)
        return sorted_results[:10]
    except Exception as e:
        logger.error(f"Error getting top ligand results: {e}")
        return []





if __name__ == "__main__":
    # Use raw strings for Windows paths
    SMILES_DIR = r'C:\Users\wes\AutoMol-v2\results\phase2a\predicted_structures'
    RESULTS_DIR = r'C:\Users\wes\AutoMol-v2\automol\results'

    high_throughput_screening()
    top_results = get_top_ligand_results()
    print("Top 10 Ligands:")
    for result in top_results:
        print(f"Name: {result['ligand_name']}, SMILES: {result['smiles']}, Score: {result['total_score']:.2f}")













