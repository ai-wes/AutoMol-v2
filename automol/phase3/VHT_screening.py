import os
import multiprocessing
from subprocess import run

from rdkit import Chem
from rdkit.Chem import Descriptors






# Step 1: Physicochemical Properties Filtering -Ligands
def filter_ligand_physicochemical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES"

    # Calculate properties
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    # Define acceptable ranges
    weight_range = (150, 500)  # molecular weight in g/mol
    logp_range = (-2, 5)  # LogP value for hydrophobicity
    tpsa_limit = 140  # Topological polar surface area in Å²

    # Filter based on ranges
    if not (weight_range[0] <= mol_weight <= weight_range[1]):
        return False, f"Molecular weight {mol_weight} out of range"
    if not (logp_range[0] <= logp <= logp_range[1]):
        return False, f"LogP {logp} out of range"
    if tpsa > tpsa_limit:
        return False, f"TPSA {tpsa} out of range"

    return True, "Ligand passes physicochemical filters"


# Step 1: Physicochemical Properties Filtering -Proteins

from Bio.SeqUtils import molecular_weight, ProtParam

def filter_protein_physicochemical(sequence):
    # Calculate molecular weight
    mol_weight = molecular_weight(sequence, seq_type="protein")

    # Use ProtParam to calculate instability index and isoelectric point
    prot_param = ProtParam.ProteinAnalysis(sequence)
    instability_index = prot_param.instability_index()
    iso_point = prot_param.isoelectric_point()

    # Define acceptable ranges
    weight_limit = 150000  # Max molecular weight in Da
    instability_limit = 40  # Instability index threshold
    iso_point_range = (4.0, 9.0)  # Acceptable range for isoelectric point

    # Filter based on properties
    if mol_weight > weight_limit:
        return False, f"Molecular weight {mol_weight} out of range"
    if instability_index > instability_limit:
        return False, f"Instability index {instability_index} too high"
    if not (iso_point_range[0] <= iso_point <= iso_point_range[1]):
        return False, f"Isoelectric point {iso_point} out of range"

    return True, "Protein passes physicochemical filters"



#Step 2: Sequence Similarity and Novelty Check -Proteins
from Bio.Blast import NCBIWWW, NCBIXML

def check_protein_similarity(sequence):
    result_handle = NCBIWWW.qblast("blastp", "nr", sequence)
    blast_record = NCBIXML.read(result_handle)

    # Check if there is a significant match (E-value < 0.05)
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            if hsp.expect < 0.05:
                return False, f"Sequence too similar to known sequence: {alignment.hit_def}"

    return True, "Sequence is sufficiently novel"



#Step 2: Novelty Check -Ligands

import requests
def check_ligand_novelty(smiles):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
    response = requests.get(url)
    if response.status_code == 200 and "IdentifierList" in response.json():
        return False, "SMILES string matches a known compound in PubChem"
    return True, "Ligand is novel"




#Step 3: Predictive Toxicity and ADMET Screening
def admet_screening(smiles):
    url = "https://biosig.lab.uq.edu.au/pkcsm/prediction"
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "smiles": smiles,
        "admet": "toxicity"
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        results = response.json()
        if results['toxicity']['alert']:
            return False, "Toxicity alert detected"
        return True, "No toxicity alerts"
    else:
        return False, "ADMET screening failed"


#Step 3: SwissADME Screening
def swiss_adme_screen(smiles):
    url = f"http://www.swissadme.ch/index.php?{smiles}"
    response = requests.get(url)
    if response.status_code == 200:
        # Parse results (this is simplified and may require more handling)
        if "No toxic" in response.text:
            return True, "No toxicity predicted by SwissADME"
        else:
            return False, "SwissADME predicts potential toxicity"
    return False, "Failed to screen with SwissADME"



def calculate_ligand_rating(smiles):
    # Step 1: Physicochemical filtering
    pass_physico, physico_message = filter_ligand_physicochemical(smiles)
    physico_score = 8 if pass_physico else 0

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

import json

def store_ligand_result(smiles, rating_details):
    results = []
    try:
        with open("ligand_screening_results.json", "r") as file:
            results = json.load(file)
    except FileNotFoundError:
        pass  # No results file yet, start with an empty list
    results.append(rating_details)
    with open("ligand_screening_results.json", "w") as file:
        json.dump(results, file, indent=4)

def screen_and_store_ligand(smiles):
    # Perform the screening and calculate the rating
    score, details = calculate_ligand_rating(smiles)

    # Store the result
    store_ligand_result(smiles, details)
    return f"Ligand {smiles} has been screened and stored with a score of {score}/10."




def get_top_ligand_results():
    try:
        with open("ligand_screening_results.json", "r") as file:
            results = json.load(file)
            # Sort by total score, descending
            sorted_results = sorted(results, key=lambda x: x['total_score'], reverse=True)
            return sorted_results[:10]  # Return top 10 results
    except FileNotFoundError:
        return "No results found."


## RUN THE PIPELINE


def screen_ligand_pipeline(smiles):
    # Step 1: Physicochemical filtering
    pass_physico, physico_message = filter_ligand_physicochemical(smiles)
    if not pass_physico:
        return f"Failed Physicochemical Filtering: {physico_message}"

    # Step 2: Novelty check
    pass_novelty, novelty_message = check_ligand_novelty(smiles)
    if not pass_novelty:
        return f"Failed Novelty Check: {novelty_message}"

    # Step 3: ADMET Screening
    pass_admet, admet_message = admet_screening(smiles)
    if not pass_admet:
        return f"Failed ADMET Screening: {admet_message}"

    return "Ligand passed all screenings and is a candidate for further inspection."


def screen_protein_pipeline(sequence):
    # Step 1: Physicochemical filtering
    pass_physico, physico_message = filter_protein_physicochemical(sequence)
    if not pass_physico:
        return f"Failed Physicochemical Filtering: {physico_message}"

    # Step 2: Novelty check (BLAST)
    pass_novelty, novelty_message = check_protein_similarity(sequence)
    if not pass_novelty:
        return f"Failed Novelty Check: {novelty_message}"

    return "Protein passed all screenings and is a candidate for further inspection."


import os
import multiprocessing
import json
from rdkit import Chem

import os
import multiprocessing
import json
from rdkit import Chem

# Use raw strings for Windows paths
SMILES_DIR = r'C:\Users\wes\AutoMol-v2\results\phase2a\predicted_structures'
RESULTS_DIR = r'C:\Users\wes\AutoMol-v2\automol\results'

def process_ligand(smiles_file):
    with open(os.path.join(SMILES_DIR, smiles_file), 'r') as f:
        smiles = f.read().strip()
    
    result = screen_ligand_pipeline(smiles)
    
    # Store the result
    ligand_name = os.path.splitext(smiles_file)[0]
    with open(os.path.join(RESULTS_DIR, f'{ligand_name}_result.json'), 'w') as f:
        json.dump(result, f, indent=4)
    
    return result

def high_throughput_screening():
    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get all SMILES files
    smiles_files = [f for f in os.listdir(SMILES_DIR) if f.endswith('.smi')]
    
    # Use multiprocessing to screen ligands in parallel
    with multiprocessing.Pool() as pool:
        results = pool.map(process_ligand, smiles_files)
    
    # Process and store overall results
    all_results = []
    for smiles_file, result in zip(smiles_files, results):
        ligand_name = os.path.splitext(smiles_file)[0]
        all_results.append({
            "ligand_name": ligand_name,
            "result": result
        })
    
    # Store all results in a single JSON file
    with open(os.path.join(RESULTS_DIR, 'all_screening_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"High-throughput screening completed. Results stored in {RESULTS_DIR}")

def get_top_ligand_results():
    try:
        with open(os.path.join(RESULTS_DIR, "all_screening_results.json"), "r") as file:
            results = json.load(file)
            # Assuming the structure is a list of dictionaries with 'result' key
            sorted_results = sorted(results, key=lambda x: x['result']['total_score'] if isinstance(x['result'], dict) and 'total_score' in x['result'] else -1, reverse=True)
            return sorted_results[:10]  # Return top 10 results
    except FileNotFoundError:
        return []  # Return an empty list if file not found
    except json.JSONDecodeError:
        print("Error decoding JSON file. It may be empty or incorrectly formatted.")
        return []

if __name__ == "__main__":
    high_throughput_screening()
    top_results = get_top_ligand_results()
    for result in top_results:
        if isinstance(result['result'], dict) and 'smiles' in result['result'] and 'total_score' in result['result']:
            print(f"SMILES: {result['result']['smiles']}, Score: {result['result']['total_score']}")
        else:
            print(f"Unexpected result structure: {result}")