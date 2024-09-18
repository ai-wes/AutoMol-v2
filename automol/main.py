# AutoMol-v2/automol/main.py

import sys
import os

import os
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + 'C:\\Users\\wes\\AutoMol-v2'

import logging
import argparse
from datetime import datetime
from phase1.phase1_run import Phase1
from phase2.phase2a.phase2a_run import run_Phase_2a
from phase2.phase2b.phase2b_run import run_Phase_2b
from phase3.phase3_run import run_Phase_3
from phase4.phase4_run import run_Phase_4  # Uncomment when Phase 4 is ready
from utils.save_utils import save_json, create_organized_directory_structure, create_phase_directory, save_results




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

def setup_logging(log_dir="logs"):
    create_organized_directory_structure(log_dir)
    log_file = os.path.join(log_dir, "autonomol.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")

def main():
    args = parse_arguments()
    setup_logging()

    logging.info("Starting AutoMol-v2 Pipeline")

    # Create base output directories if they don't exist
    if not os.path.exists(args.output_dir):
        create_organized_directory_structure(args.output_dir)
    
    phase1_dir = os.path.join(args.output_dir, "phase1")
    phase2a_dir = os.path.join(args.output_dir, "phase2a")
    phase2b_dir = os.path.join(args.output_dir, "phase2b")
    phase3_dir = os.path.join(args.output_dir, "phase3")
    phase4_dir = os.path.join(args.output_dir, "phase4")
    
    for directory in [phase1_dir, phase2a_dir, phase2b_dir, phase3_dir, phase4_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    try:
        # Phase 1: Technical Description Generation
        phase1 = Phase1()
        if not args.skip_description_gen:
            descriptions = phase1.run_pipeline(args.input_text)
            logging.info("Phase 1 completed: Technical descriptions generated.")
            # Save Phase 1 outputs
            save_json({"technical_descriptions": descriptions}, os.path.join(phase1_dir, "descriptions.json"))
        else:
            descriptions = [args.input_text]
            logging.info("Phase 1 skipped: Using input text directly.")
            save_json({"technical_descriptions": descriptions}, os.path.join(phase1_dir, "descriptions.json"))

        # Phase 2a: Protein Generation and Optimization
        logging.info("Starting Phase 2a: Protein Generation and Optimization.")
        analysis_results_phase2a, protein_sequences = run_Phase_2a(
            technical_descriptions=descriptions,
            predicted_structures_dir=os.path.join(phase2a_dir, "predicted_structures"),
            results_dir=phase2a_dir,
            num_sequences=args.num_sequences,
            optimization_steps=args.optimization_steps,
            score_threshold=args.score_threshold
        )
        logging.info("Phase 2a completed: Proteins generated and optimized.")
        # Save Phase 2a outputs
        save_json({
            "analysis_results_phase2a": analysis_results_phase2a,
            "protein_sequences": protein_sequences
        }, os.path.join(phase2a_dir, "phase2a_results.json"))

        # Phase 2b: Ligand Generation and Optimization
        logging.info("Starting Phase 2b: Ligand Generation and Optimization.")
        ligand_results = run_Phase_2b(
            predicted_structures_dir=os.path.join(phase2a_dir, "predicted_structures"),
            results_dir=phase2b_dir,
            num_sequences=args.num_sequences,
            optimization_steps=args.optimization_steps,
            score_threshold=args.score_threshold,
            protein_sequences=protein_sequences  # Pass the generated protein sequences
        )
        logging.info("Phase 2b completed: Ligands generated and optimized.")
        # Save Phase 2b outputs
        save_json({
            "ligand_results": ligand_results
        }, os.path.join(phase2b_dir, "phase2b_results.json"))

        # Phase 3: Analysis
        logging.info("Starting Phase 3: Analysis.")
        phase3_results = run_Phase_3(
            protein_results=analysis_results_phase2a,
            ligand_results=ligand_results,
            input_text=args.input_text,
            output_dir=phase3_dir
        )
        logging.info("Phase 3 completed: Analysis performed.")
        # Save Phase 3 outputs
        save_json({
            "phase3_results": phase3_results
        }, os.path.join(phase3_dir, "phase3_results.json"))

        # Phase 4: Virtual Lab Simulation and Automation
        # Uncomment and implement Phase 4 when ready
        logging.info("Starting Phase 4: Virtual Lab Simulation and Automation.")
        simulation_results = run_Phase_4(phase3_results)
        logging.info("Phase 4 completed: Virtual lab simulation and automation done.")
        save_json({
            "simulation_results": simulation_results
        }, os.path.join(phase4_dir, "phase4_results.json"))

        # Save final results
        logging.info("Saving final results.")
        final_results = {
            "phase1": os.path.join(phase1_dir, "descriptions.json"),
            "phase2a": os.path.join(phase2a_dir, "phase2a_results.json"),
            "phase2b": os.path.join(phase2b_dir, "phase2b_results.json"),
            "phase3": os.path.join(phase3_dir, "phase3_results.json"),
            # "phase4": os.path.join(phase4_dir, "phase4_results.json")
        }
        save_results(final_results, args.output_dir)
        logging.info("AutoMol-v2 Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred in the AutoMol-v2 Pipeline: {e}", exc_info=True)
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
