import os
import asyncio
import logging
from Phase_2_protein.analyze import run_analysis_pipeline
from Phase_2_protein.generate import generate_protein_sequence
from Phase_2_protein.optimize_new import run_optimization_pipeline
from Phase_2_protein.predict import run_prediction_pipeline
from Phase_2_protein.simulate import run_simulation_pipeline
from utils.save_utils import create_sequence_directories, save_partial_results
from utils.shared_state import set_protein_sequences


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_Phase_2(input_text, optimization_steps, score_threshold, technical_descriptions, predicted_structures_dir, results_dir):
    logger.info("\nStarting Phase 2: Generating and analyzing novel proteins")

    all_analysis_results = []
    best_score = 0
    all_generated_sequences = []

    for i, desc in enumerate(technical_descriptions):
        technical_instruction = desc['technical_instruction']
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

            generated_sequence = generated_sequences[0]
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
                            attempts += 1
                            continue

                        prediction_result = prediction_results[0]
                        pdb_file = prediction_result['pdb_file']

                        simulation_result = await run_simulation_pipeline(pdb_file, simulation_dir)

                        analysis_result = await run_analysis_pipeline(
                            simulation_result['trajectory_file'],
                            simulation_result['final_pdb'],
                            analysis_dir
                        )

                        if analysis_result:
                            analysis_result['sequence'] = optimized_sequence
                            analysis_result['score'] = analysis_result['final_score']
                            analysis_result['optimization_info'] = {
                                'optimized_score': optimized_score,
                                'best_method': best_method
                            }
                            analysis_result['technical_description'] = technical_instruction
                            all_analysis_results.append(analysis_result)
                            best_score = optimized_score
                            logger.info(f"New best score: {best_score}")
                            await save_partial_results(all_analysis_results, results_dir, i)
                            break
                        else:
                            logger.info(f"Skipping simulation for sequence with score {optimized_score} (not better than {best_score})")

            attempts += 1
            if attempts == max_attempts:
                logger.info(f"Reached maximum attempts ({max_attempts}) for Technical Description {i+1}. Moving to next description.")

    await set_protein_sequences(all_generated_sequences)

    return all_analysis_results