# AutoMol-v2/automol/phase3/run_Phase_3.py

import os
import sys
import logging
import gc
import torch

from multiprocessing import cpu_count

# Importing the separate modules
from phase3.simulate import run_simulation_pipeline
from phase3.docking_simulation import run_docking_pipeline
from phase3.pymol_analysis import run_pymol_analysis_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phase3_run.log")
    ]
)
logger = logging.getLogger(__name__)

def run_Phase_3(protein_results, ligand_results, input_text, output_dir):
    """
    Run Phase 3 including simulation, docking, and PyMOL analysis.
    
    Parameters:
    - protein_results: List of protein result dictionaries.
    - ligand_results: List of ligand result dictionaries.
    - input_text: Additional input parameters as needed.
    - output_dir: Directory to store all outputs.
    
    Returns:
    - Dictionary containing paths to simulation, docking, and analysis results.
    """
    try:
        # Ensure output directories exist
        simulation_dir = os.path.join(output_dir, "simulation")
        docking_dir = os.path.join(output_dir, "docking")
        analysis_dir = os.path.join(output_dir, "analysis")
        
        for directory in [simulation_dir, docking_dir, analysis_dir]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created/Verified directory: {directory}")
        
        # Determine device for simulation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for simulation: {device}")
        

        # Step 2: Docking Simulation
        logger.info("Starting Docking Simulation...")
        docking_results = run_docking_pipeline(protein_results, ligand_results, docking_dir)
        logger.info("Docking Simulation completed.")
        
        # Step 1: Molecular Dynamics Simulation
        logger.info("Starting Molecular Dynamics Simulation...")
        simulation_results = run_simulation_pipeline(protein_results, simulation_dir, device)
        logger.info("Molecular Dynamics Simulation completed.")
        
        # Step 3: PyMOL Analysis
        logger.info("Starting PyMOL Analysis...")
        analysis_results = run_pymol_analysis_pipeline(simulation_results, analysis_dir)
        logger.info("PyMOL Analysis completed.")
        
        phase3_results = {
            "simulation_dir": simulation_dir,
            "docking_dir": docking_dir,
            "analysis_dir": analysis_dir,
            "simulation_results": simulation_results,
            "docking_results": docking_results,
            "analysis_results": analysis_results
        }
        
        return phase3_results
    
    except Exception as e:
        logger.error(f"Error in Phase 3: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Phase 3 run completed. Resources cleaned up.")

if __name__ == "__main__":
    # Example usage (Replace with actual data retrieval logic)
    protein_results = [
        {"id": 1, "pdb_file": "path/to/protein1.pdb"},
        {"id": 2, "pdb_file": "path/to/protein2.pdb"}
    ]
    
    ligand_results = [
        {"id": 1, "pdbqt_file": "path/to/ligand1.pdbqt"},
        {"id": 2, "pdbqt_file": "path/to/ligand2.pdbqt"}
    ]
    
    input_text = "Additional input parameters if any"
    output_dir = os.path.join(os.getcwd(), "results", "phase3")
    
    results = run_phase_3(protein_results, ligand_results, input_text, output_dir)
    logger.info(f"Phase 3 Results: {results}")
