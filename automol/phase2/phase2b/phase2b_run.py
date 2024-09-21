# AutoMol-v2/automol/phase2/phase2b/phase2b_run.py
import json
import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from colorama import Fore, Style, init

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
) -> List[Dict[str, Any]]:
    try:
        print(Fore.CYAN + "Running Phase 2b: SMILES Ligand Pipeline")
        logger.info("Running Phase 2b: SMILES Ligand Pipeline")
        
        pipeline = SMILESLigandPipeline()
        
        results = pipeline.run_smiles_ligand_pipeline(
            predicted_structures_dir=predicted_structures_dir,
            results_dir=results_dir,
            num_sequences=num_sequences,
            optimization_steps=optimization_steps,
            score_threshold=score_threshold,
            protein_sequences=protein_sequences
        )
        
        processed_results = []
        for i, result in enumerate(results):
            smiles = result.get('smiles', '')
            score = result.get('score', 0)
            
            processed_result = {
                'smiles': smiles,
                'score': score
            }
            processed_results.append(processed_result)
            
            print(Fore.GREEN + f"Processed result {i+1}:")
            print(Fore.GREEN + f"  SMILES: {smiles}")
            print(Fore.GREEN + f"  Score: {score}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"phase2b_results_{timestamp}.json"
        results_path = os.path.join(results_dir, results_filename)
        
        with open(results_path, "w") as f:
            json.dump(processed_results, f, indent=4)
        
        print(Fore.GREEN + f"Phase 2b completed successfully with {len(processed_results)} results.")
        print(Fore.GREEN + f"Results saved to: {results_path}")
        logger.info(f"Phase 2b completed successfully with {len(processed_results)} results.")
        logger.info(f"Results saved to: {results_path}")
        return processed_results
    except Exception as e:
        print(Fore.RED + f"An error occurred in run_Phase_2b: {e}")
        logger.error(f"An error occurred in run_Phase_2b: {e}")
        raise

def main():
    predicted_structures_dir = "predicted_structures"
    results_dir = "results/phase2b"
    num_sequences = 2
    optimization_steps = 100
    score_threshold = 0.8

    # Ensure output directories exist
    Path(predicted_structures_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    try:
        ligand_sequences = run_Phase_2b(
            predicted_structures_dir=predicted_structures_dir,
            results_dir=results_dir,
            num_sequences=num_sequences,
            optimization_steps=optimization_steps,
            score_threshold=score_threshold,
            protein_sequences=[]  # This should be populated from Phase 2a
        )
        print(Fore.CYAN + f"Generated ligand sequences: {ligand_sequences}")
    except Exception as e:
        print(Fore.RED + f"An error occurred: {str(e)}")
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
        main()
