import json
import sys
from dotenv import load_dotenv
load_dotenv()
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from colorama import Fore, Style, init
import torch

# Initialize colorama
init(autoreset=True)

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from phase2b.SMILESLigandPipeline import SMILESLigandPipeline
from phase2b.pre_screen_compounds import pre_screen_ligand

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

pipeline = SMILESLigandPipeline()

def run_Phase_2b(
    predicted_structures_dir: str,
    results_dir: str,
    num_sequences: int,
    optimization_steps: int,
    score_threshold: float,
    protein_sequences: List[str]
) -> Dict[str, Any]:
    logger.info("Starting Phase 2b: Generate and Optimize Ligands")
    
    phase2b_results = []
    
    for protein_sequence in protein_sequences:
        logger.info(f"Processing protein sequence: {protein_sequence}")
        novel_smiles = pipeline.generate_novel_smiles(protein_sequence, num_sequences)
        valid_ligands = []
        max_optimization_attempts = 10  # Maximum number of optimization attempts

        for smiles in novel_smiles:
            passed, message = pre_screen_ligand(smiles)  # Validation of ligand
            attempts = 0

            while not passed and attempts < max_optimization_attempts:
                logger.warning(f"Ligand {smiles} failed validation: {message}. Attempting optimization. Attempt {attempts + 1}/{max_optimization_attempts}")
                print(f"WARNING: Ligand {smiles} failed validation: {message}. Attempting optimization. Attempt {attempts + 1}/{max_optimization_attempts}")
                smiles = pipeline.iterative_optimization(smiles)
                passed, message = pre_screen_ligand(smiles)
                attempts += 1
                logger.info(f"Attempt {attempts}: Ligand {smiles} validation status: {passed}, message: {message}")
                print(f"Attempt {attempts}: Ligand {smiles} validation status: {passed}, message: {message}")

            if passed:
                valid_ligands.append({"smiles": smiles})
                logger.info(f"Ligand {smiles} passed validation after {attempts} attempts: {message}")
                print(f"Ligand {smiles} passed validation after {attempts} attempts: {message}")
            else:
                logger.warning(f"Ligand {smiles} failed validation after {attempts} attempts: {message}")
                print(f"WARNING: Ligand {smiles} failed validation after {attempts} attempts: {message}")

        if not valid_ligands:
            logger.error("No ligands passed validation. Skipping this protein sequence.")
            print("ERROR: No ligands passed validation. Skipping this protein sequence.")
            continue

        # Process docking for valid ligands
        for ligand in valid_ligands:
            try:
                docking_result = pipeline.process_single_smiles(
                    smiles=ligand['smiles'],
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

# Function to execute Phase 2b with example inputs
def main():
    predicted_structures_dir = 'example_predicted_structures'
    results_dir = 'example_results'
    num_sequences = 3
    optimization_steps = 5
    score_threshold = -8.0
    protein_sequences = ['MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYVELKSGAGSS']
    
    print("Protein sequences: ", protein_sequences)
    result = run_Phase_2b(
        predicted_structures_dir=predicted_structures_dir,
        results_dir=results_dir,
        num_sequences=num_sequences,
        optimization_steps=optimization_steps,
        score_threshold=score_threshold,
        protein_sequences=protein_sequences
    )
    print("Result: ", result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()