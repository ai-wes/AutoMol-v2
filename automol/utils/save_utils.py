import os
from datetime import datetime
import csv
import asyncio
import json

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

async def save_results_to_csv(input_text, sequences, scores, timestamp, filename, **kwargs):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Input Text', 'Sequence', 'Score', 'Timestamp'] + list(kwargs.keys()))
        for i, (seq, score) in enumerate(zip(sequences, scores)):
            row = [input_text, seq, score, timestamp]
            row.extend([kwargs[key][i] for key in kwargs])
            writer.writerow(row)

async def save_partial_results(results, run_dir, description_index):
    if results:
        partial_results_file = os.path.join(run_dir, f'partial_results_description_{description_index}.csv')

        final_sequences = []
        final_scores = []
        optimization_scores = []
        optimization_methods = []
        technical_descriptions = []

        for result in results:
            final_sequences.append(result['sequence'])
            final_scores.append(result['score'])
            optimization_scores.append(result['optimization_info']['optimized_score'])
            optimization_methods.append(result['optimization_info']['best_method'])
            technical_descriptions.append(result['technical_description'])

        await save_results_to_csv(
            f"Partial results for description {description_index}", 
            final_sequences, 
            final_scores, 
            datetime.now().strftime("%Y%m%d_%H%M%S"), 
            partial_results_file, 
            optimization_scores=optimization_scores,
            optimization_methods=optimization_methods,
            technical_descriptions=technical_descriptions
        )
        return partial_results_file
    return None

def save_results(output_dir, analysis_results):
    """
    Save the analysis results to various formats in the specified output directory.
    
    Args:
    output_dir (str): The directory to save the results.
    analysis_results (dict): The analysis results to be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_filename = os.path.join(output_dir, f"analysis_results_{timestamp}.json")
    with open(json_filename, 'w') as json_file:
        json.dump(analysis_results, json_file, indent=2)
    
    # Save as CSV
    csv_filename = os.path.join(output_dir, f"analysis_results_{timestamp}.csv")
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Key', 'Value'])  # Header
        for key, value in analysis_results.items():
            writer.writerow([key, value])
    
    # Save as text
    txt_filename = os.path.join(output_dir, f"analysis_results_{timestamp}.txt")
    with open(txt_filename, 'w') as txt_file:
        for key, value in analysis_results.items():
            txt_file.write(f"{key}: {value}\n")
    
    print(f"Results saved in {output_dir}:")
    print(f"- JSON: {json_filename}")
    print(f"- CSV: {csv_filename}")
    print(f"- Text: {txt_filename}")