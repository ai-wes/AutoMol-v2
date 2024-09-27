# AutoMol-v2/automol/server/utils/save_utils.py

import os
from datetime import datetime
import csv
import json
import logging

def create_organized_directory_structure(base_output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    # Define phase-specific directories
    phase_dirs = {
        "phase1": ["processed_articles"],
        "phase2a": ["generated_sequences"],
        "phase2b": ["ligands"],
        "phase3": ["simulations", "docking", "analysis"],
        "phase4": [],
        "logs": []  # Add a logs directory
    }
    
    # Create directories
    for phase, subdirs in phase_dirs.items():
        phase_dir = os.path.join(run_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(phase_dir, subdir), exist_ok=True)
    
    log_file_path = os.path.join(run_dir, "logs", "pipeline.log")
    return run_dir, phase_dirs, log_file_path

def create_sequence_directories(results_dir, sequence_id):
    """
    Creates directories for analysis and simulation for a given sequence.
    """
    analysis_dir = os.path.join(results_dir, f"structure_{sequence_id}", "analysis")
    simulation_dir = os.path.join(results_dir, f"structure_{sequence_id}", "simulation")

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(simulation_dir, exist_ok=True)

    return analysis_dir, simulation_dir

def save_json(data, filepath):
    """
    Saves a dictionary to a JSON file.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved JSON data to: {filepath}")
    except Exception as e:
        logging.error(f"Error saving JSON data to {filepath}: {e}")
        raise

def load_json(filepath):
    """
    Loads a dictionary from a JSON file.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded JSON data from: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON data from {filepath}: {e}")
        raise

def save_results_csv(results, filepath):
    """
    Saves results to a CSV file.
    """
    try:
        keys = results[0].keys()
        with open(filepath, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        logging.info(f"Saved CSV results to: {filepath}")
    except Exception as e:
        logging.error(f"Error saving CSV results to {filepath}: {e}")
        raise

def save_results(data, base_output_dir):
    """
    Saves all results to a JSON file in the base output directory.
    """
    try:
        save_json(data, os.path.join(base_output_dir, "final_results.json"))
    except Exception as e:
        logging.error(f"Error saving final results: {e}")
        raise