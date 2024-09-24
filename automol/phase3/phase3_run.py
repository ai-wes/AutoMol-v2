# AutoMol-v2/automol/phase3/run_Phase_3.py

import os
import sys
# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import colorama
from colorama import Fore, Back, Style, init
import logging
import gc
import torch
from openbabel import openbabel



import logging
import os
import gc
import torch
from typing import List, Dict, Any

from multiprocessing import cpu_count

# Importing the separate modules
from phase3.simulate import run_simulation_pipeline
from phase3.docking_simulation import run_docking_pipeline
from phase3.analyze import run_analysis_pipeline

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


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"{Fore.GREEN}CUDA cache cleared.")
        logging.info("CUDA cache cleared.")
    else:
        print(f"{Fore.YELLOW}CUDA is not available. Skipping CUDA cache clearing.")
        logging.info("CUDA is not available. Skipping CUDA cache clearing.")




def convert_pdb_to_pdbqt(input_pdb, output_pdbqt):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_pdb)
    
    obConversion.WriteFile(mol, output_pdbqt)
    return output_pdbqt


def run_Phase_3(
    protein_results: List[Dict[str, Any]],
    simulation_dir: str,
    device: str
) -> Dict[str, Any]:
    """
    Run Phase 3 including simulation, docking, and analysis.
    Returns a dictionary with simulation, docking, and analysis results.
    """
    logger = logging.getLogger(__name__)
    try:
        # Ensure output directories exist
        simulation_subdir = os.path.join(simulation_dir, "simulations")
        docking_dir = os.path.join(simulation_dir, "docking")
        analysis_dir = os.path.join(simulation_dir, "analysis")
        
        for directory in [simulation_subdir, docking_dir, analysis_dir]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created/Verified directory: {directory}")
        
        # Determine device for simulation
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for simulation: {device}")

        # Convert ligand PDB files to PDBQT
        logger.info("Converting ligand PDB files to PDBQT format...")
        for ligand in protein_results:
            if 'predicted_pdb' in ligand:
                ligand_id = ligand.get('id', 'unknown')
                protein_sim_dir = os.path.join(simulation_subdir, f"protein_{ligand_id}")
                os.makedirs(protein_sim_dir, exist_ok=True)
                pdb_file = ligand.get('predicted_pdb')
                pdbqt_file = convert_pdb_to_pdbqt(pdb_file, protein_sim_dir)
                ligand['pdbqt_file'] = pdbqt_file
                logger.info(f"Converted {pdb_file} to {pdbqt_file}")
        
        # Docking Simulation
        logger.info("Starting Docking Simulation...")
        docking_results = run_docking_pipeline(protein_results, protein_results, docking_dir)  # Pass all required arguments
        logger.info("Docking Simulation completed.")
        
        # Molecular Dynamics Simulation
        logger.info("Starting Molecular Dynamics Simulation...")
        simulation_results = run_simulation_pipeline(protein_results, simulation_subdir, device)
        logger.info("Molecular Dynamics Simulation completed.")
        
        # Analysis
        logger.info("Starting Analysis...")
        analysis_results = run_analysis_pipeline(simulation_results, protein_results, analysis_dir, device)
        logger.info("Analysis completed.")
        
        phase3_results = {
            "simulation_results": simulation_results,
            "docking_results": docking_results,
            "analysis_results": analysis_results
        }
        logger.info("Phase 3 results compiled.")
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