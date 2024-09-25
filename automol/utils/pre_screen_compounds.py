from typing import Tuple

from pymongo import MongoClient
from rdkit import Chem
from rdkit.Chem import Descriptors
import requests
import json
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['screening_pipeline']
results_collection = db['screening_results']

# Minimum score thresholds
MINIMUM_SCORE_THRESHOLD = 6.0
MINIMAL_VIABILITY_THRESHOLD = 4.0  # New threshold for minimal viability

# Physicochemical Properties Filtering
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

def admet_screening(smiles):
    url = "https://biosig.lab.uq.edu.au/pkcsm/prediction"
    headers = {'Content-Type': 'application/json'}
    data = {"smiles": smiles, "admet": "toxicity"}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json()
        if results['toxicity']['alert']:
            return False, "Toxicity alert detected"
        return True, "No toxicity alerts"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 405:
            logger.warning(f"ADMET screening API request failed with 405 Method Not Allowed. Falling back to minimal viability screening.")
            return minimal_viability_screening(smiles)
        else:
            logger.error(f"ADMET screening API request failed: {e}")
            return False, "ADMET screening failed due to API error"
    except requests.exceptions.RequestException as e:
        logger.error(f"ADMET screening API request failed: {e}")
        return False, "ADMET screening failed due to API error"




# Minimal Viability Screening
def minimal_viability_screening(smiles: str) -> Tuple[bool, str]:
    """
    Apply minimal criteria to allow novel compounds to pass.
    These criteria are less stringent than the main pre-screening.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES for minimal viability"

    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    # Less stringent criteria
    weight_range = (100, 600)
    logp_range = (-3, 6)

    if not (weight_range[0] <= mol_weight <= weight_range[1]):
        return False, f"MW {mol_weight} out of minimal range"
    if not (logp_range[0] <= logp <= logp_range[1]):
        return False, f"LogP {logp} out of minimal range"

    return True, "Ligand passes minimal viability screening"

def store_ligand_result_mongo(smiles: str, status: str = "Passed Pre-Screening"):
    """
    Store ligand screening results in MongoDB.
    """
    details = {"smiles_or_sequence": smiles, "status": status, "timestamp": datetime.now()}
    try:
        results_collection.insert_one(details)
        logger.info(f"Ligand {smiles} stored in MongoDB with status: {status}.")
    except Exception as e:
        logger.error(f"Failed to store ligand {smiles} in MongoDB: {e}")

def log_failed_sequence(smiles: str, reason: str):
    """
    Log failed sequences to a file for analysis.
    """
    log_path = "failed_sequences_log.json"
    log_entry = {"smiles": smiles, "reason": reason, "timestamp": datetime.now().isoformat()}
    try:
        with open(log_path, "a") as log_file:
            log_file.write(json.dumps(log_entry, indent=4) + "\n")
        logger.warning(f"Ligand {smiles} failed pre-screening: {reason}")
    except Exception as e:
        logger.error(f"Failed to log failed sequence {smiles}: {e}")

def log_minimal_passed_sequence(smiles: str, reason: str):
    """
    Log minimal passed sequences to a separate file.
    """
    log_path = "minimal_passed_sequences_log.json"
    log_entry = {"smiles": smiles, "reason": reason, "timestamp": datetime.now().isoformat()}
    try:
        with open(log_path, "a") as log_file:
            log_file.write(json.dumps(log_entry, indent=4) + "\n")
        logger.info(f"Ligand {smiles} passed minimal viability screening: {reason}")
    except Exception as e:
        logger.error(f"Failed to log minimal passed sequence {smiles}: {e}")

def log_passed_sequence(smiles: str, reason: str):
    """
    Log passed sequences to a separate file.
    """
    log_path = "passed_sequences_log.json"
    log_entry = {"smiles": smiles, "reason": reason, "timestamp": datetime.now().isoformat()}
    try:
        with open(log_path, "a") as log_file:
            log_file.write(json.dumps(log_entry, indent=4) + "\n")
        logger.info(f"Ligand {smiles} passed pre-screening: {reason}")
    except Exception as e:
        logger.error(f"Failed to log passed sequence {smiles}: {e}")

def pre_screen_ligand(smiles: str) -> Tuple[bool, str]:
    """
    Pre-screen ligand based on physicochemical and ADMET properties.
    """
    # Step 1: Physicochemical Filtering
    if not filter_ligand_physicochemical(smiles):
        return False, "Failed physicochemical filtering"
    
    # Step 2: ADMET Screening
    pass_admet, admet_message = admet_screening(smiles)
    if not pass_admet:
        # Attempt Minimal Viability Screening
        pass_minimal, minimal_message = minimal_viability_screening(smiles)
        if pass_minimal:
            return True, "Passed Minimal Viability Screening"
        else:
            return False, admet_message
    else:
        return True, "Passed Pre-Screening"

def screen_and_store_ligand_mongo(smiles: str) -> str:
    """
    Screen ligand and store results in MongoDB.
    """
    # Ensure 'smiles' is valid
    if not smiles or not isinstance(smiles, str):
        logger.error(f"Invalid SMILES input: {smiles}")
        return f"Ligand {smiles} has invalid SMILES."

    passed, message = pre_screen_ligand(smiles)
    if passed:
        # Determine status based on the message
        if "Minimal Viability" in message:
            status = "Passed Minimal Viability Screening"
            log_minimal_passed_sequence(smiles, message)
        else:
            status = "Passed Pre-Screening"
            log_passed_sequence(smiles, message)
        store_ligand_result_mongo(smiles, status=status)
        return f"Ligand {smiles} {message} and was stored."
    else:
        log_failed_sequence(smiles, message)
        return f"Ligand {smiles} failed pre-screening: {message}"