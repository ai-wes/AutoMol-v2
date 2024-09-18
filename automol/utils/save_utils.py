import os
from datetime import datetime
import csv
import asyncio
import json
    
# AutoMol-v2/automol/utils/save_utils.py

import os
import csv
import json
import logging
def create_organized_directory_structure(base_output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")

    # Create subdirectories
    predicted_structures_dir = os.path.join(run_dir, "predicted_structures")
    results_dir = os.path.join(run_dir, "results")

    os.makedirs(predicted_structures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return run_dir, predicted_structures_dir, results_dir

def create_sequence_directories(results_dir, sequence_id):
    analysis_dir = os.path.join(results_dir, f"structure_{sequence_id}", "analysis")
    simulation_dir = os.path.join(results_dir, f"structure_{sequence_id}", "simulation")

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(simulation_dir, exist_ok=True)

    return analysis_dir, simulation_dir

    
    


def create_organized_directory_structure(base_dir):
    try:
        os.makedirs(base_dir, exist_ok=True)
        logging.info(f"Created directory structure at: {base_dir}")
    except Exception as e:
        logging.error(f"Error creating directory structure: {e}")
        raise

def create_phase_directory(base_dir, phase_name):
    phase_dir = os.path.join(base_dir, phase_name)
    create_organized_directory_structure(phase_dir)
    return phase_dir

def save_json(data, filepath):
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved JSON data to: {filepath}")
    except Exception as e:
        logging.error(f"Error saving JSON data to {filepath}: {e}")
        raise

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded JSON data from: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON data from {filepath}: {e}")
        raise

def save_results_csv(results, filepath):
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
    try:
        save_json(data, os.path.join(base_output_dir, "final_results.json"))
    except Exception as e:
        logging.error(f"Error saving final results: {e}")
        raise
