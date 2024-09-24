import shutil
import sys
import os
import logging
from colorama import Fore, Style, init
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Initialize colorama
init(autoreset=True)

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from phase2a.generate import generate_protein_sequence
from phase2a.optimize_new import run_optimization_pipeline
from phase2a.predict import run_prediction_pipeline
from utils.save_utils import create_sequence_directories
from phase2a.shared_state import set_protein_sequences

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_Phase_2a(
    technical_descriptions: List[str],
    predicted_structures_dir: str,
    results_dir: str,
    num_sequences: int,
    optimization_steps: int,
    score_threshold: float
) -> Tuple[List[Dict[str, Any]], List[str]]:
    print(Fore.CYAN + "\nStarting Phase 2a: Generating and Optimizing novel proteins")
    logger.info("Starting Phase 2a: Generating and Optimizing novel proteins")

    all_analysis_results = []
    best_score = 0
    all_generated_sequences = []

    for i, desc in enumerate(technical_descriptions):
        technical_instruction = str(desc)
        print(Fore.YELLOW + f"\nProcessing Technical Description {i+1}:")
        print(Fore.WHITE + f"- {technical_instruction}")
        logger.info(f"Processing Technical Description {i+1}: {technical_instruction}")

        attempts = 0
        max_attempts = 2

        while attempts < max_attempts:
            try:
                generated_sequences = generate_protein_sequence(technical_instruction, num_sequences)
                if not generated_sequences:
                    print(Fore.RED + f"Failed to generate sequence for attempt {attempts + 1}. Retrying...")
                    logger.warning(f"Failed to generate sequence for attempt {attempts + 1}. Retrying...")
                    attempts += 1
                    continue

                # Select the top sequence based on some criteria, e.g., the first one
                generated_sequence = generated_sequences[0]
                all_generated_sequences.append(generated_sequence)
                print(Fore.GREEN + f"Generated sequence (attempt {attempts + 1}): {generated_sequence[:50]}...")
                logger.info(f"Generated sequence (attempt {attempts + 1}): {generated_sequence[:50]}...")

                optimized_results = run_optimization_pipeline(
                    [generated_sequence],
                    iterations=optimization_steps,
                    score_threshold=score_threshold
                )

                if optimized_results:
                    for opt_result in optimized_results:
                        optimized_sequence = opt_result.get('optimized_sequence')
                        optimized_score = opt_result.get('optimized_score', 0)
                        best_method = opt_result.get('best_method', 'N/A')

                        print(Fore.BLUE + f"Setting Initial Protein Sequences: {all_generated_sequences}...")
                        set_protein_sequences(sequences=[optimized_sequence], scores=[optimized_score], score_threshold=score_threshold)
                        print(Fore.BLUE + "Protein sequences have been set in the shared state.")
                        logger.info("Protein sequences have been set in the shared state.")

                        print(Fore.CYAN + f"Optimized Score: {optimized_score}")
                        print(Fore.CYAN + f"Best Method: {best_method}")
                        logger.debug(f"Optimized Sequence: {optimized_sequence}")
                        logger.debug(f"Optimized Score: {optimized_score}")
                        logger.debug(f"Best Method: {best_method}")

                        if optimized_score > best_score:
                            best_score = optimized_score
                            analysis_dir, simulation_dir = create_sequence_directories(
                                results_dir, len(all_analysis_results)
                            )
                            prediction_results = run_prediction_pipeline(
                                [optimized_sequence],
                                output_dir=predicted_structures_dir
                            )

                            if not prediction_results or not prediction_results[0].get('pdb_file'):
                                print(Fore.RED + "Prediction failed. Skipping simulation and analysis.")
                                logger.error("Prediction failed. Skipping simulation and analysis.")
                                attempts += 1
                                continue

                            pdb_file = prediction_results[0]['pdb_file']
                            pdb_filename = os.path.basename(pdb_file)
                            new_pdb_path = os.path.join(analysis_dir, pdb_filename)
                            shutil.copy(pdb_file, new_pdb_path)

                            analysis_result = {
                                'sequence': optimized_sequence,
                                'score': optimized_score,
                                'pdb_file': new_pdb_path,
                                'analysis_dir': analysis_dir
                            }
                            all_analysis_results.append(analysis_result)        
                            print(Fore.GREEN + f"Analysis completed for sequence with score {optimized_score}.")
                            logger.info(f"Analysis completed for sequence with score {optimized_score}.")

                attempts += 1

            except Exception as e:
                print(Fore.RED + f"An error occurred during Phase 2a processing: {e}")
                logger.error(f"An error occurred during Phase 2a processing: {e}", exc_info=True)
                attempts += 1

            if attempts == max_attempts:
                print(Fore.YELLOW + f"Reached maximum attempts ({max_attempts}) for Technical Description {i+1}. Moving to next description.")
                logger.info(f"Reached maximum attempts ({max_attempts}) for Technical Description {i+1}. Moving to next description.")

    # Removed internal saving
    set_protein_sequences(sequences=all_generated_sequences, scores=[1.0] * len(all_generated_sequences), score_threshold=score_threshold)
    print(Fore.GREEN + "Phase 2a completed: All protein sequences generated and analyzed.")
    logger.info("Phase 2a completed: All protein sequences generated and analyzed.")

    return all_analysis_results, all_generated_sequences