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
    print("Starting Phase 2b: Generate and Optimize Ligands")
    phase2b_results = []
    
    total_proteins = len(protein_sequences)
    for idx, protein_sequence in enumerate(protein_sequences):
        progress = int((idx / total_proteins) * 100)
        logger.info(f"Processing protein sequence: {protein_sequence}")
        novel_smiles = pipeline.generate_valid_novel_smiles(protein_sequence, num_sequences)
        valid_ligands = []
        max_optimization_attempts = 10  # Maximum number of optimization attempts
        print(f"Processing protein sequence {idx+1}/{total_proteins}")
        
        total_smiles = len(novel_smiles)
        for smiles_idx, smiles in enumerate(novel_smiles):
            smiles_progress = progress + int((smiles_idx / total_smiles) * (100 / total_proteins))
            passed, message = pre_screen_ligand(smiles)  # Validation of ligand
            attempts = 0
            print("Validating SMILES {smiles_idx+1}/{total_smiles}")
            while not passed and attempts < max_optimization_attempts:
                logger.warning(f"Ligand {smiles} failed validation: {message}. Attempting optimization. Attempt {attempts + 1}/{max_optimization_attempts}")
                print(f"WARNING: Ligand {smiles} failed validation: {message}. Attempting optimization. Attempt {attempts + 1}/{max_optimization_attempts}")
                smiles = pipeline.optimizer.optimize_smiles(smiles)
                passed, message = pre_screen_ligand(smiles)
                attempts += 1
                logger.info(f"Attempt {attempts}: Ligand {smiles} validation status: {passed}, message: {message}")
                print(f"Attempt {attempts}: Ligand {smiles} validation status: {passed}, message: {message}")
                print("Optimizing SMILES, attempt {attempts}/{max_optimization_attempts}")
            if passed:
                valid_ligands.append({"smiles": smiles})
                logger.info(f"Ligand {smiles} passed validation after {attempts} attempts: {message}")
                print(f"Ligand {smiles} passed validation after {attempts} attempts: {message}")
                print("SMILES {smiles_idx+1}/{total_smiles} passed validation")
            else:
                logger.warning(f"Ligand {smiles} failed validation after {attempts} attempts: {message}")
                print(f"WARNING: Ligand {smiles} failed validation after {attempts} attempts: {message}")
                print("SMILES {smiles_idx+1}/{total_smiles} failed validation")
        if not valid_ligands:
            logger.error("No ligands passed validation. Skipping this protein sequence.")
            print("ERROR: No ligands passed validation. Skipping this protein sequence.")
            print("No valid ligands for current protein sequence")
            continue

        # Process docking for valid ligands
        total_valid_ligands = len(valid_ligands)
        for ligand_idx, ligand in enumerate(valid_ligands):
            ligand_progress = progress + int((ligand_idx / total_valid_ligands) * (100 / total_proteins))
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
                print(f"Error in docking process: {str(e)}")
                continue
    
    logger.info("Phase 2b completed successfully.")
    print("Phase 2b completed successfully")
    return {"phase2b_results": phase2b_results}

# Function to execute Phase 2b with example inputs
def main():
    predicted_structures_dir = 'example_predicted_structures'
    results_dir = 'example_results'
    num_sequences = 3
    optimization_steps = 5
    score_threshold = -8.0
    protein_sequences = ['MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYVELKSGAGSS']
    
    print("Starting Phase 2b execution")
    result = run_Phase_2b(
        predicted_structures_dir=predicted_structures_dir,
        results_dir=results_dir,
        num_sequences=num_sequences,
        optimization_steps=optimization_steps,
        score_threshold=score_threshold,
        protein_sequences=protein_sequences
    )
    print("Phase 2b execution completed")

if __name__ == "__main__":
    main()