# AutoMol-v2/automol/main.py

import sys
import os
import logging
import asyncio
import argparse
from datetime import datetime
from phase1.phase1_run import Phase1
from phase2.phase2a.phase2a_run import run_Phase_2a
from phase2.phase2b.phase2b_run import run_Phase_2b
from phase3.run_Phase_3 import run_Phase_3
from utils.save_utils import (
    create_organized_directory_structure,
    create_sequence_directories,
    save_results_to_csv,
    save_partial_results,
    save_results
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="AutoMol-v2: Novel molecule generation and analysis pipeline")
    parser.add_argument("--input_text", type=str, required=True, help="Input text describing the desired molecule function")
    parser.add_argument("--max_generations", type=int, default=2, help="Maximum number of generations for technical descriptions")
    parser.add_argument("--num_reflections", type=int, default=2, help="Number of reflection rounds for technical descriptions")
    parser.add_argument("--num_sequences", type=int, default=2, help="Number of molecule sequences to generate initially")
    parser.add_argument("--optimization_steps", type=int, default=15, help="Number of optimization steps to perform")
    parser.add_argument("--score_threshold", type=float, default=0.55, help="Minimum score threshold for accepting generated sequences")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--skip_description_gen", action="store_true", help="Skip technical description generation and use input text directly")
    return parser.parse_args()

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
    
    predicted_structures_dir = "predicted_structures"
    results_dir = "results"
    
    # Phase 2: Molecule Generation and Optimization
    protein_task =  run_Phase_2a(
            descriptions,
            predicted_structures_dir,
            results_dir,
            args.num_sequences,
            args.optimization_steps,
            args.score_threshold
        )
    
    ligand_task = run_Phase_2b(
            descriptions,
            predicted_structures_dir,
            results_dir,
            args.num_sequences,
            args.optimization_steps,
            args.score_threshold
        )
    
    
    proteins, ligands = await asyncio.gather(protein_task, ligand_task)
    logging.info("Phase 2 completed: Molecules generated and optimized.")
    
    # Phase 3: Analysis
    analysis_results = run_Phase_3(proteins, ligands)
    logging.info("Phase 3 completed: Analysis performed.")
    
    # Phase 4: Virtual Lab Simulation and Automation
    # simulation_results = run_Phase_4(analysis_results)
    # logging.info("Phase 4 completed: Virtual lab simulation and automation done.")
    
    # Save results
    save_results(args.output_dir, analysis_results)
    logging.info("AutoMol-v2 Pipeline completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())