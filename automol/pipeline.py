import logging
import os
import sys
import json
from pathlib import Path
from typing import Callable, Dict, Any
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from phase1.phase1_run import run_Phase_1
from phase2.phase2a.phase2a_run import run_Phase_2a
from phase2.phase2b.phase2b_run import run_Phase_2b
from phase3.phase3_run import run_Phase_3
from phase4.phase4_run import run_Phase_4
from phase5.phase5_run import run_Phase_5
from utils.save_utils import save_json, create_organized_directory_structure

logger = logging.getLogger(__name__)

def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge command-line arguments into the configuration dictionary."""
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    return config

def run_main_pipeline(config: Dict[str, Any], emit_progress: Callable[[str, int, str], None]) -> None:
    """Execute the entire AutoMol-v2 pipeline."""
    try:
        emit_progress("Phase 1 - Research and Hypothesize", 0, "Starting AutoMol-v2 pipeline...")

        # Create organized directory structure
        emit_progress("Phase 1 - Research and Hypothesize", 15, "Creating organized directory structure...")
        try:
            run_dir, phase_dirs, log_file_path = create_organized_directory_structure(config['base_output_dir'])
            if not run_dir or not phase_dirs:
                raise ValueError("Failed to create directory structure")
            logger.info(f"Organized directory structure created at {run_dir}.")
            emit_progress("Phase 1 - Research and Hypothesize", 20, f"Organized directory structure created at {run_dir}.")
        except Exception as e:
            logger.error(f"Failed to create directory structure: {str(e)}")
            emit_progress("Phase 1 - Research and Hypothesize", 0, f"Failed to create directory structure: {str(e)}")
            sys.exit(1)

        # Update config with run_dir
        config['run_dir'] = run_dir
        emit_progress("Phase 1 - Research and Hypothesize", 25, "Updated config with run_dir.")

        try:
            if config.get('skip_description_gen', False):
                phase1_results = {
                    'technical_description': config['input_text']
                }
                save_json(phase1_results, Path(run_dir) / "phase1_results.json")
                logger.info("Phase 1 results saved successfully.")
                emit_progress("Phase 1 - Skipped Research and Hypothesize", 100, "Phase 1 results saved successfully.")
            else:
                # Phase 1: Generate Hypothesis
                logger.info("Starting Phase 1: Generate Hypothesis")
                emit_progress("Phase 1 - Research and Hypothesize", 30, "Starting Phase 1: Generate Hypothesis")
                phase1_results = run_Phase_1(config['phase1'])
                save_json(phase1_results, Path(run_dir) / "phase1_results.json")
                logger.info("Phase 1 results saved successfully.")
                emit_progress("Phase 1 - Research and Hypothesize", 40, "Phase 1 results saved successfully.")
            
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
            emit_progress("Phase 2 - Experimentation and Data Collection", 50, "Phase 2a results saved successfully.")

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
            emit_progress("Phase 2 - Experimentation and Data Collection", 75, "Phase 2b results saved successfully.")

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
            emit_progress("Phase 3 - Analysis and Interpretation", 90, "Phase 3 results saved successfully.")

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
            emit_progress("Phase 4 - Validation and Verification", 100, "Phase 4 results saved successfully.")

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
            emit_progress("Phase 4 - Validation and Verification", 100, "All phase results saved successfully.")

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
            emit_progress("Phase 5 - Final Report and Decision Making Process", 100, "Phase 5 results saved successfully.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            emit_progress("System", 0, f"An unexpected error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        emit_progress("System", 0, f"An unexpected error occurred: {e}")