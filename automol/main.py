# src/main.py
import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now import predict from the same directory

import logging
from phase1.phase1_run import Phase1
from phase2.phase2a.phase2a_run import run_Phase_2a
from phase2.phase2b.phase2b_run import SMILESLigandPipeline
from phase3.run_Phase_3 import run_Phase_3
import asyncio
import argparse
import asyncio
from datetime import datetime
import logging
from utils.save_utils import create_organized_directory_structure, create_sequence_directories, save_results_to_csv, save_partial_results, save_results
# Add these imports



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="AutoMol-v2: Novel molecule generation and analysis pipeline")
    parser.add_argument("--input_text", type=str, required=True,
                        help="Input text describing the desired molecule function")
    parser.add_argument("--max_generations", type=int, default=2,
                        help="Maximum number of generations for technical descriptions")
    parser.add_argument("--num_reflections", type=int, default=2,
                        help="Number of reflection rounds for technical descriptions")
    parser.add_argument("--num_sequences", type=int, default=2,
                        help="Number of molecule sequences to generate initially")
    parser.add_argument("--optimization_steps", type=int, default=15,
                        help="Number of optimization steps to perform")
    parser.add_argument("--score_threshold", type=float, default=0.55,
                        help="Minimum score threshold for accepting generated sequences")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--skip_description_gen", action="store_true",
                        help="Skip technical description generation and use input text directly")


    return parser.parse_args()

async def run_autoprot_framework():
    args = parse_arguments()
    start_time = datetime.now()

    logger.info(f"Starting AutoProt Framework at {start_time}")
    logger.info(f"User Prompt: {args.input_text}")

    # Create the main run directory
    run_dir, predicted_structures_dir, results_dir = create_organized_directory_structure(args.output_dir)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("autonomol.log"),
            logging.StreamHandler()
        ]
    )
    
phase1 = Phase1()

async def main():
    args = parse_arguments()
    setup_logging()
    logging.info("Starting AutoMol-v2 Pipeline")
    # Phase 1: Technical Description Generation
    if not args.skip_description_gen:
        descriptions = phase1.run_pipeline(args.input_text)
    else:
        descriptions = [args.input_text]
    logging.info("Phase 1 completed: Technical descriptions generated.")
    phase2b = SMILESLigandPipeline()

    # Phase 2: Molecule Generation and Optimization
    protein_task = asyncio.create_task(run_Phase_2a(descriptions, args.num_sequences, args.optimization_steps, args.score_threshold))
    ligand_task = asyncio.create_task(phase2b.run_Phase_2b(descriptions, args.num_sequences, args.optimization_steps, args.score_threshold))
    proteins, ligands = await asyncio.gather(protein_task, ligand_task)
    logging.info("Phase 2 completed: Molecules generated and optimized.")

    # Phase 3: Analysis
    analysis_results = run_Phase_3(proteins, ligands)
    logging.info("Phase 3 completed: Analysis performed.")

    # Phase 4: Virtual Lab Simulation and Automation
   # simulation_results = run_Phase_4(analysis_results)
    #logging.info("Phase 4 completed: Virtual lab simulation and automation done.")

    # Save results
    save_results(args.output_dir, analysis_results)

    logging.info("AutoMol-v2 Pipeline completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())