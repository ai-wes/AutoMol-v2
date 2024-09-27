# AutoMol-v2/automol/server/main.py

import os
from datetime import datetime
import json
import sys
import logging
from pathlib import Path
import subprocess
import argparse
import socketio  # Import Socket.IO client

from phase1.phase1_run import run_Phase_1
from phase2.phase2a.phase2a_run import run_Phase_2a
from phase2.phase2b.phase2b_run import run_Phase_2b
from phase3.phase3_run import run_Phase_3
from phase4.phase4_run import run_Phase_4
from phase5.phase5_run import run_Phase_5
from utils.save_utils import save_json, create_organized_directory_structure
from server.app import emit_progress, socketio

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

# Initialize Socket.IO client



def merge_config_with_args(config, args):
    """Merge command-line arguments into the configuration dictionary."""
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="AutoMol-v2: Novel molecule generation and analysis pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--input_text", type=str, help="Input text describing the desired molecule function")
    parser.add_argument("--num_sequences", type=int, help="Number of molecule sequences to generate initially")
    parser.add_argument("--optimization_steps", type=int, help="Number of optimization steps to perform")
    parser.add_argument("--score_threshold", type=float, help="Minimum score threshold for accepting generated sequences")
    parser.add_argument("--device", type=str, help="Device to use for computations (cuda or cpu)")
    parser.add_argument("--skip_description_gen", action="store_true", help="Skip the description generation phase")
    return parser.parse_args()

def run_main_pipeline():
    logger = logging.getLogger(__name__)

    args = parse_arguments()

    emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message="Starting AutoMol-v2 pipeline...")

    # Load configuration
    script_dir = Path(__file__).parent
    CONFIG_PATH = Path(__file__).parent.parent / args.config
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config = json.load(config_file)
        logger.info("Configuration loaded successfully.")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=5, message="Configuration loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Config file not found at {CONFIG_PATH}.")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message=f"Config file not found at {CONFIG_PATH}.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message="Invalid JSON in config file.")
        sys.exit(1)

    # Update config with command-line arguments (only if provided)
    config = merge_config_with_args(config, args)

    logger.info("Configuration merged with command-line arguments.")
    emit_progress(phase="Phase 1 - Research and Hypothesize", progress=10, message="Configuration merged with command-line arguments.")


    # Ensure all required keys are present in the config
    required_keys = [
        'base_output_dir', 'input_text', 'num_sequences', 'optimization_steps',
        'score_threshold', 'device', 'output_paths', 'phase1', 'phase2a',
        'phase2b', 'phase3', 'phase4', 'mongodb'
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"Missing required configuration key(s): {', '.join(missing_keys)}")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message=f"Missing required configuration key(s): {', '.join(missing_keys)}")
        socketio.disconnect()
        sys.exit(1)

    # Set up logging after verifying base_output_dir
    log_file = Path(config['base_output_dir']) / config['output_paths']['log_file']
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create log directory at {log_file.parent}: {e}")
        emit_progress(phase="System", progress=0, message=f"Failed to create log directory: {e}")
        socketio.disconnect()
        sys.exit(1)

    # Reconfigure logging to include the new log file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger(__name__)

    # Create organized directory structure
    emit_progress(phase="Phase 1 - Research and Hypothesize", progress=15, message="Creating organized directory structure...")
    try:
        run_dir, phase_dirs, log_file_path = create_organized_directory_structure(config['base_output_dir'])
        if not run_dir or not phase_dirs:
            raise ValueError("Failed to create directory structure")
        logger.info(f"Organized directory structure created at {run_dir}.")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=20, message=f"Organized directory structure created at {run_dir}.")
    except Exception as e:
        logger.error(f"Failed to create directory structure: {str(e)}")
        emit_progress(phase="Phase 1 - Research and Hypothesize", progress=0, message=f"Failed to create directory structure: {str(e)}")
        socketio.disconnect()
        sys.exit(1)

    # Update config with run_dir
    config['run_dir'] = run_dir
    emit_progress(phase="Phase 1 - Research and Hypothesize", progress=25, message="Updated config with run_dir.")

    try:
        if config.get('skip_description_gen', False):
            phase1_results = {
                'technical_description': config['input_text']
            }
            save_json(phase1_results, Path(run_dir) / "phase1_results.json")
            logger.info("Phase 1 results saved successfully.")
            emit_progress(phase="Phase 1 - Skipped Research and Hypothesize", progress=100, message="Phase 1 results saved successfully.")
        else:
            # Phase 1: Generate Hypothesis
            logger.info("Starting Phase 1: Generate Hypothesis")
            emit_progress(phase="Phase 1 - Research and Hypothesize", progress=30, message="Starting Phase 1: Generate Hypothesis")
            phase1_results = run_Phase_1(config['phase1'])
            save_json(phase1_results, Path(run_dir) / "phase1_results.json")
            logger.info("Phase 1 results saved successfully.")
            emit_progress(phase="Phase 1 - Research and Hypothesize", progress=40, message="Phase 1 results saved successfully.")
        
        # Phase 2a: Generate and Optimize Proteins
        logger.info("Starting Phase 2a: Generate and Optimize Proteins")
        phase2a_config = config['phase2a']
        phase2a_config.update({
            'technical_descriptions': [phase1_results['technical_description']],
            'predicted_structures_dir': os.path.join(run_dir, "phase2a", "generated_sequences"),
            'results_dir': os.path.join(run_dir, "phase2a", "results"),
            'num_sequences': config['num_sequences'],
            'optimization_steps': config['optimization_steps'],
            'score_threshold': config['score_threshold']
        })
        phase2a_results, all_generated_sequences = run_Phase_2a(**phase2a_config)
        save_json(phase2a_results, Path(run_dir) / "phase2a_results.json")
        logger.info("Phase 2a results saved successfully.")
        emit_progress(phase="Phase 2 - Experimentation and Data Collection", progress=50, message="Phase 2a results saved successfully.")

        # Extract protein sequences from phase2a_results
        protein_sequences = [result['sequence'] for result in phase2a_results]

        # Phase 2b: Generate and Optimize Ligands
        logger.info("Starting Phase 2b: Generate and Optimize Ligands")
        phase2b_config = config['phase2b']
        phase2b_config.update({
            'predicted_structures_dir': os.path.join(run_dir, "phase2b", "ligands"),
            'results_dir': os.path.join(run_dir, "phase2b", "results"),
            'num_sequences': config['num_sequences'],
            'optimization_steps': config['optimization_steps'],
            'score_threshold': config['score_threshold'],
            'protein_sequences': protein_sequences
        })
        phase2b_results = run_Phase_2b(**phase2b_config)
        save_json(phase2b_results, Path(run_dir) / "phase2b_results.json")
        logger.info("Phase 2b results saved successfully.")
        emit_progress(phase="Phase 2 - Experimentation and Data Collection", progress=75, message="Phase 2b results saved successfully.")

        # Phase 3: Simulation
        logger.info("Starting Phase 3: Simulation")
        phase3_config = config['phase3']
        phase3_config.update({
            'protein_results': phase2a_results,
            'ligand_results': phase2b_results,
            'output_dir': os.path.join(run_dir, "phase3"),
            'device': config['device']
        })
        phase3_results = run_Phase_3(**phase3_config)
        save_json(phase3_results, Path(run_dir) / "phase3" / "phase3_results.json")
        logger.info("Phase 3 results saved successfully.")
        emit_progress(phase="Phase 3 - Analysis and Interpretation", progress=90, message="Phase 3 results saved successfully.")

        # Phase 4: Final Analysis and Reporting
        logger.info("Starting Phase 4: Final Analysis and Reporting")
        phase4_config = config['phase4']
        phase4_config.update({
            'simulation_results': phase3_results,
            'output_dir': os.path.join(run_dir, "phase4")
        })
        phase4_results = run_Phase_4(phase3_results, phase4_config)
        save_json(phase4_results, Path(run_dir) / "phase4_results.json")
        logger.info("Phase 4 results saved successfully.")
        emit_progress(phase="Phase 4 - Validation and Verification", progress=100, message="Phase 4 results saved successfully.")

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
        emit_progress(phase="Phase 4 - Validation and Verification", progress=100, message="All phase results saved successfully.")

        # Phase 5: Final Report and Decision Making Process
        logger.info("Starting Phase 5: Decision Making Process")
        phase5_config = config['phase5']
        base_output_dir = config['base_output_dir']
        phase5_config.update({
            'base_output_dir': base_output_dir  
        })
        phase5_results = run_Phase_5(phase5_config)
        save_json(phase5_results, Path(run_dir) / "phase5_results.json")
        logger.info("Phase 5 results saved successfully.")
        emit_progress(phase="Phase 5 - Final Report and Decision Making Process", progress=100, message="Phase 5 results saved successfully.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        emit_progress(phase="System", progress=0, message=f"An unexpected error occurred: {e}")
    finally:
        socketio.disconnect()

if __name__ == "__main__":
    main()