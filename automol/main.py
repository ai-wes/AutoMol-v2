import json
import sys
import logging
from pathlib import Path
import os
from phase1.phase1_run import run_Phase_1
from phase2.phase2a.phase2a_run import run_Phase_2a
from phase2.phase2b.phase2b_run import run_Phase_2b
from phase3.phase3_run import run_Phase_3
from phase4.phase4_run import run_Phase_4
from utils.save_utils import save_json, create_organized_directory_structure
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("main_run.log")
    ]
)
logger = logging.getLogger(__name__)

def merge_config_with_args(config, args):
    """Merge command-line arguments into the configuration dictionary."""
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config

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

def main():
    # Load configuration
    args = parse_arguments()
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logger.error("config.json file not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config.json.")
        sys.exit(1)

    # Merge config with arguments
    config = merge_config_with_args(config, args)
    logger.info("Configuration merged with command-line arguments.")

    config_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(config_overrides)
    logger.info(f"Configuration overridden with command-line arguments: {config_overrides}")

    # Create organized directory structure
    base_output_dir = config.get('base_output_dir', 'results')
    try:
        run_dir, phase_dirs = create_organized_directory_structure(base_output_dir)
        if not run_dir or not phase_dirs:
            raise ValueError("Failed to create directory structure")
        logger.info(f"Organized directory structure created at {run_dir}.")
    except Exception as e:
        logger.error(f"Failed to create directory structure: {str(e)}")
        sys.exit(1)

    # Update config with run_dir
    config['run_dir'] = run_dir

    # Phase 1: Generate Hypothesis
    if not config.get('skip_description_gen', False):
        logger.info("Starting Phase 1: Generate Hypothesis")
        phase1_results = run_Phase_1(config)
        save_json(phase1_results, Path(run_dir) / "phase1_results.json")
        logger.info("Phase 1 results saved successfully.")
    else:
        logger.info("Skipping Phase 1: Generate Hypothesis as per command-line flag.")
        # Optionally, set default values or handle accordingly
        phase1_results = {
            'technical_description': config.get('input_text', ''),
            'initial_hypothesis': config.get('input_text', '')
        }
    print("phase1_results", phase1_results)

    # Phase 2a: Generate and Optimize Proteins
    logger.info("Starting Phase 2a: Generate and Optimize Proteins")
    phase2a_config = config['phase2a']
    phase2a_config.update({
        'technical_descriptions': [phase1_results['technical_description']],  # Ensure it's a list
        'predicted_structures_dir': os.path.join(run_dir, "phase2a", "generated_sequences"),
        'results_dir': os.path.join(run_dir, "phase2a", "results"),
        'num_sequences': config.get('num_sequences', 2),
        'optimization_steps': config.get('optimization_steps', 15),
        'score_threshold': config.get('score_threshold', 0.55)
    })

    # Filter out unexpected keys
    expected_keys = ['technical_descriptions', 'predicted_structures_dir', 'results_dir', 'num_sequences', 'optimization_steps', 'score_threshold']
    filtered_phase2a_config = {k: v for k, v in phase2a_config.items() if k in expected_keys}

    # Unpack the filtered configuration dictionary when calling run_Phase_2a
    phase2a_results, all_generated_sequences = run_Phase_2a(**filtered_phase2a_config)
    save_json(phase2a_results, Path(run_dir) / "phase2a_results.json")
    logger.info("Phase 2a results saved successfully.")

    # Extract protein sequences from phase2a_results
    protein_sequences = [result['sequence'] for result in phase2a_results]

    # Phase 2b: Generate and Optimize Ligands
    # Phase 2b: Generate and Optimize Ligands
    logger.info("Starting Phase 2b: Generate and Optimize Ligands")
    phase2b_config = config['phase2b']
    phase2b_config.update({
        'predicted_structures_dir': os.path.join(run_dir, "phase2b", "ligands"),  # Corrected path
        'results_dir': os.path.join(run_dir, "phase2b", "results"),
        'num_sequences': config.get('num_sequences', 2),
        'optimization_steps': config.get('optimization_steps', 15),
        'score_threshold': config.get('score_threshold', 0.55),
        'protein_sequences': protein_sequences
    })

    # Filter out unexpected keys
    expected_keys_phase2b = ['predicted_structures_dir', 'results_dir', 'num_sequences', 'optimization_steps', 'score_threshold', 'protein_sequences']
    filtered_phase2b_config = {k: v for k, v in phase2b_config.items() if k in expected_keys_phase2b}

    phase2b_results = run_Phase_2b(**filtered_phase2b_config)
    save_json(phase2b_results, Path(run_dir) / "phase2b_results.json")
    logger.info("Phase 2b results saved successfully.")


    # Phase 3: Simulation
    logger.info("Starting Phase 3: Simulation")
    phase3_config = config['phase3']
    phase3_config.update({
        'protein_results': phase2a_results,
        'output_dir': os.path.join(run_dir, "phase3"),
        'device': config.get('device', 'cpu')
    })

    # Optionally, filter out unexpected keys for Phase 3
    expected_keys_phase3 = ['protein_results', 'output_dir', 'device']
    filtered_phase3_config = {k: v for k, v in phase3_config.items() if k in expected_keys_phase3}

    # Unpack the filtered configuration dictionary when calling run_Phase_3
    phase3_results = run_Phase_3(**filtered_phase3_config)
    save_json(phase3_results, Path(run_dir) / "phase3" / "phase3_results.json")
    logger.info("Phase 3 results saved successfully.")


    # Phase 4: Final Analysis and Reporting
    # Phase 4: Final Analysis and Reporting
    logger.info("Starting Phase 4: Final Analysis and Reporting")
    phase4_config = config['phase4']
    phase4_config.update({
        'simulation_results': phase3_results,
        'output_dir': os.path.join(run_dir, "phase4")
    })

    # Pass both phase3_results and config to run_Phase_4
    phase4_results = run_Phase_4(phase3_results, phase4_config)
    save_json(phase4_results, Path(run_dir) / "phase4_results.json")
    logger.info("Phase 4 results saved successfully.")


    # Save All Results Consolidated
    all_results = {
        'phase1': phase1_results,
        'phase2a': phase2a_results,
        'phase2b': phase2b_results,
        'phase3': phase3_results,
        'phase4': phase4_results
    }
    save_json(all_results, Path(run_dir) / "final_results.json")
    logger.info("All phase results saved successfully.")

if __name__ == "__main__":
    main()