# AutoMol-v2/automol/phase2/phase2b/phase2b_run.py
import json
import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from colorama import Fore, Style, init
import torch
from utils.pre_screen_compounds import pre_screen_ligand
# Initialize colorama
init(autoreset=True)

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from phase2b.SMILESLigandPipeline import SMILESLigandPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_Phase_2b(
    predicted_structures_dir: str,
    results_dir: str,
    num_sequences: int,
    optimization_steps: int,
    score_threshold: float,
    protein_sequences: List[str]
) -> Dict[str, Any]:
    logger.info("Starting Phase 2b: Generate and Optimize Ligands")
    
    pipeline = SMILESLigandPipeline()
    phase2b_results = []
    
    for protein_sequence in protein_sequences:
        logger.info(f"Processing protein sequence: {protein_sequence}")
        print(Fore.CYAN + f"Processing protein sequence: {protein_sequence}")
        
        # Generate novel SMILES
        try:
            novel_smiles = pipeline.generate_novel_smiles(protein_sequence, num_sequences)
            logger.info(f"Generated {len(novel_smiles)} novel SMILES.")
            print(Fore.GREEN + f"Generated {len(novel_smiles)} novel SMILES.")
        except Exception as e:
            logger.error(f"Error generating novel SMILES: {e}")
            print(Fore.RED + f"Error generating novel SMILES: {e}")
            continue
        valid_ligands = []
        max_optimization_attempts = 10  # Maximum number of optimization attempts

        for ligand in results:
            # Check if 'smiles' key exists
            if 'smiles' not in ligand:
                logging.warning(f"Ligand entry missing 'smiles' key: {ligand}")
                print(f"WARNING: Ligand entry missing 'smiles' key: {ligand}")
                continue  # Skip this entry

            smiles = ligand['smiles']
            passed, message = pre_screen_ligand(smiles)  # Validation of ligand
            attempts = 0

            while not passed and attempts < max_optimization_attempts:
                logging.warning(f"Ligand {smiles} failed validation: {message}. Attempting optimization. Attempt {attempts + 1}/{max_optimization_attempts}")
                print(f"WARNING: Ligand {smiles} failed validation: {message}. Attempting optimization. Attempt {attempts + 1}/{max_optimization_attempts}")
                smiles = pipeline.iterative_optimization(smiles)  # Corrected argument count
                passed, message = pre_screen_ligand(smiles)
                attempts += 1
                logging.info(f"Attempt {attempts}: Ligand {smiles} validation status: {passed}, message: {message}")
                print(f"Attempt {attempts}: Ligand {smiles} validation status: {passed}, message: {message}")

            if passed:
                ligand['smiles'] = smiles
                valid_ligands.append(ligand)
                logging.info(f"Ligand {smiles} passed validation after {attempts} attempts: {message}")
                print(f"Ligand {smiles} passed validation after {attempts} attempts: {message}")
            else:
                logging.warning(f"Ligand {smiles} failed validation after {attempts} attempts: {message}")
                print(f"WARNING: Ligand {smiles} failed validation after {attempts} attempts: {message}")

        if not valid_ligands:
            logging.error("No ligands passed validation. Exiting pipeline.")
            print("ERROR: No ligands passed validation. Exiting pipeline.")
            return []  # Return an empty list instead of None        

        for smiles in novel_smiles:
            # Pre-screen and store ligand
            try:
                storage_message = pre_screen_ligand(smiles)
                logger.info(storage_message)
                print(Fore.GREEN + storage_message)
            except Exception as e:
                logger.error(f"Error during pre-screening: {e}")
                print(Fore.RED + f"Error during pre-screening: {e}")
                continue
            
            # Process docking
            try:
                docking_result = pipeline.process_single_smiles(
                    smiles=smiles,
                    protein_sequence=protein_sequence,
                    predicted_structures_dir=predicted_structures_dir,
                    results_dir=results_dir,
                    score_threshold=score_threshold
                )
                if docking_result:
                    phase2b_results.append(docking_result)
            except Exception as e:
                logger.error(f"Error during docking process: {e}")
                print(Fore.RED + f"Error during docking process: {e}")
                continue
    
    logger.info("Phase 2b completed successfully.")
    print(Fore.GREEN + "Phase 2b completed successfully.")
    return {"phase2b_results": phase2b_results}
