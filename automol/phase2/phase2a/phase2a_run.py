#set the path to the parent directory
import sys
sys.path.append('..')

import os
import asyncio
import logging

import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now import predict from the same directory
from generate import generate_protein_sequence
from optimize_new import run_optimization_pipeline
from predict import run_prediction_pipeline
from automol.utils.save_utils import create_sequence_directories, save_partial_results
from automol.utils.shared_state import set_protein_sequences


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_Phase_2a(technical_descriptions, predicted_structures_dir, results_dir, num_sequences, optimization_steps, score_threshold):
    logger.info("\nStarting Phase 2: Generating and analyzing novel proteins")

    all_analysis_results = []
    best_score = 0
    all_generated_sequences = []

    # Check if technical_descriptions is iterable
    if not isinstance(technical_descriptions, (list, tuple)):
        logger.error(f"Invalid technical_descriptions: expected list or tuple, got {type(technical_descriptions)}")
        return all_analysis_results

    for i, technical_instruction in enumerate(technical_descriptions):
        logger.info(f"\nProcessing Technical Description {i+1}:")
        logger.info(f"- {technical_instruction}")

        attempts = 0
        max_attempts = 2

        while attempts < max_attempts:
            generated_sequences = await generate_protein_sequence(technical_instruction)
            if not generated_sequences:
                logger.warning(f"Failed to generate sequence for attempt {attempts + 1}. Retrying...")
                attempts += 1
                continue

            for generated_sequence in generated_sequences[:num_sequences]:
                all_generated_sequences.append(generated_sequence)
                logger.info(f"Generated sequence (attempt {attempts + 1}): {generated_sequence[:50]}...")

                optimized_results = await run_optimization_pipeline([generated_sequence], iterations=optimization_steps, score_threshold=score_threshold)
                if optimized_results:
                    logger.info(f"Optimized {len(optimized_results)} sequences")

                    for opt_result in optimized_results:
                        optimized_sequence = opt_result['optimized_sequence']
                        optimized_score = opt_result['optimized_score']
                        best_method = opt_result['best_method']

                        if optimized_score > best_score:
                            analysis_dir, simulation_dir = create_sequence_directories(results_dir, len(all_analysis_results))

                            prediction_results = await run_prediction_pipeline([optimized_sequence], output_dir=predicted_structures_dir)
                            
                            if not prediction_results or not prediction_results[0]['pdb_file']:
                                logger.error(f"Prediction failed. Skipping simulation and analysis.")
                                continue

                            prediction_result = prediction_results[0]
                            pdb_file = prediction_result['pdb_file']

                            # TODO: Add analysis and simulation steps here
                            # analysis_results = ...
                            # all_analysis_results.append(analysis_results)

            attempts += 1
            if attempts == max_attempts:
                logger.info(f"Reached maximum attempts ({max_attempts}) for Technical Description {i+1}. Moving to next description.")

    await set_protein_sequences(all_generated_sequences)

    return all_analysis_results

# Example usage
async def main():
    technical_descriptions = [
        "Design a protein that can bind to glucose with high affinity",
        "Create a thermostable enzyme for breaking down cellulose"
    ]
    predicted_structures_dir = "predicted_structures"
    results_dir = "results"
    num_sequences = 5
    optimization_steps = 100
    score_threshold = 0.8

    try:
        protein_sequences = await run_Phase_2a(
            technical_descriptions,
            predicted_structures_dir,
            results_dir,
            num_sequences,
            optimization_steps,
            score_threshold
        )
        print(f"Generated protein sequences: {protein_sequences}")
        return protein_sequences
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())