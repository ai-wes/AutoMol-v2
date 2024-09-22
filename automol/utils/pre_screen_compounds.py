import pymongo
from pymongo import MongoClient
from rdkit import Chem
from rdkit.Chem import Descriptors
from Bio.SeqUtils import molecular_weight, ProtParam
from Bio.Blast import NCBIWWW, NCBIXML
import requests
import json

# Initialize MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['screening_pipeline']
results_collection = db['screening_results']

# Minimum score threshold for passing pre-screening
MINIMUM_SCORE_THRESHOLD = 7.0

# Step 1: Physicochemical Properties Filtering
def filter_ligand_physicochemical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES"

    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    weight_range = (150, 500)
    logp_range = (-2, 5)
    tpsa_limit = 140

    if not (weight_range[0] <= mol_weight <= weight_range[1]):
        return False, f"Molecular weight {mol_weight} out of range"
    if not (logp_range[0] <= logp <= logp_range[1]):
        return False, f"LogP {logp} out of range"
    if tpsa > tpsa_limit:
        return False, f"TPSA {tpsa} out of range"

    return True, "Ligand passes physicochemical filters"

# Step 2: Novelty Check for Ligands (PubChem Search)
def check_ligand_novelty(smiles):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
    response = requests.get(url)
    if response.status_code == 200 and "IdentifierList" in response.json():
        return False, "SMILES string matches a known compound in PubChem"
    return True, "Ligand is novel"

# Step 3: Predictive Toxicity and ADMET Screening (Mock API Call)
def admet_screening(smiles):
    url = "https://biosig.lab.uq.edu.au/pkcsm/prediction"
    headers = {'Content-Type': 'application/json'}
    data = {"smiles": smiles, "admet": "toxicity"}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        results = response.json()
        if results['toxicity']['alert']:
            return False, "Toxicity alert detected"
        return True, "No toxicity alerts"
    else:
        return False, "ADMET screening failed"

# Pre-Screening Function (Runs Validation Steps 1-3)
def pre_screen_ligand(smiles):
    # Step 1: Physicochemical Properties Filtering
    pass_physico, physico_message = filter_ligand_physicochemical(smiles)
    if not pass_physico:
        log_failed_sequence(smiles, physico_message)
        return False, physico_message

    # Step 2: Novelty Check
    pass_novelty, novelty_message = check_ligand_novelty(smiles)
    if not pass_novelty:
        log_failed_sequence(smiles, novelty_message)
        return False, novelty_message

    # Step 3: ADMET Screening
    pass_admet, admet_message = admet_screening(smiles)
    if not pass_admet:
        log_failed_sequence(smiles, admet_message)
        return False, admet_message

    return True, "Ligand passed pre-screening"

# Store result in MongoDB (only if passed pre-screening)
def store_ligand_result_mongo(smiles):
    details = {"smiles_or_sequence": smiles, "status": "Passed Pre-Screening"}
    results_collection.insert_one(details)
    print(f"Ligand {smiles} stored in MongoDB.")

# Log failed sequences to a file for analysis
def log_failed_sequence(smiles, reason):
    with open("failed_sequences_log.json", "a") as log_file:
        log_file.write(json.dumps({"smiles": smiles, "reason": reason}, indent=4) + "\n")
    print(f"Ligand {smiles} failed pre-screening: {reason}")

# Main function execution with pre-screening
def screen_and_store_ligand_mongo(smiles):
    passed, message = pre_screen_ligand(smiles)
    if passed:
        store_ligand_result_mongo(smiles)
        return f"Ligand {smiles} passed pre-screening and was stored."
    else:
        return f"Ligand {smiles} failed pre-screening: {message}"

if __name__ == "__main__":
    # Example input SMILES
    smiles_input = "CCO"
    print(screen_and_store_ligand_mongo(smiles_input))
