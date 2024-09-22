# AutoMol-v2/automol/phase3/run_Phase_3.py

import os
import sys
# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging
import gc
import torch
from openbabel import openbabel

from multiprocessing import cpu_count

# Importing the separate modules
from phase3.simulate import run_simulation_pipeline
from phase3.docking_simulation import run_docking_pipeline

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



def convert_pdb_to_pdbqt(pdb_file, output_dir):
    """
    Convert a PDB file to PDBQT format using OpenBabel.
    
    Parameters:
    - pdb_file: Path to the input PDB file
    - output_dir: Directory to save the output PDBQT file
    
    Returns:
    - Path to the output PDBQT file
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, pdb_file)
    
    pdbqt_file = os.path.join(output_dir, os.path.splitext(os.path.basename(pdb_file))[0] + ".pdbqt")
    obConversion.WriteFile(mol, pdbqt_file)
    
    return pdbqt_file




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
        


        # Convert ligand PDB files to PDBQT
        logger.info("Converting ligand PDB files to PDBQT format...")
        for ligand in ligand_results:
            if 'pdb_file' in ligand:
                ligand['pdbqt_file'] = convert_pdb_to_pdbqt(ligand['pdb_file'], docking_dir)
                logger.info(f"Converted {ligand['pdb_file']} to {ligand['pdbqt_file']}")
            else:
                logger.warning(f"No PDB file found for ligand {ligand['id']}")

        # Step 2: Docking Simulation
        logger.info("Starting Docking Simulation...")
        docking_results = run_docking_pipeline(protein_results, ligand_results, docking_dir)
        logger.info("Docking Simulation completed.")



        # Step 1: Molecular Dynamics Simulation
        logger.info("Starting Molecular Dynamics Simulation...")
        simulation_results = run_simulation_pipeline(protein_results, simulation_dir, device)
        logger.info("Molecular Dynamics Simulation completed.")
        

        
        phase3_results = {
            "simulation_dir": simulation_dir,
            "docking_dir": docking_dir,
            "analysis_dir": analysis_dir,
            "simulation_results": simulation_results,
            "docking_results": docking_results,
            "analysis_results": "Analysis results here"
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
        {"id": 1, "pdb_file": r"C:\Users\wes\AutoMol-v2\automol\predicted_structures\TNHTMIGING_structure.pdb"},
        {"id": 2, "pdb_file": r"C:\Users\wes\AutoMol-v2\automol\predicted_structures\TYFKGCIIIA_structure.pdb"}
    ]
    
    ligand_results = [
        {"id": 1, "pdb_file": r"C:\Users\wes\AutoMol-v2\results\phase2a\predicted_structures\ligand_20240917_224337.pdb"},
        {"id": 2, "pdb_file": r"C:\Users\wes\AutoMol-v2\results\phase2a\predicted_structures\ligand_20240918_003202.pdb"}
    ]
    
    input_text = "Additional input parameters if any"
    output_dir = os.path.join(os.getcwd(), "results", "phase3")
    
    results = run_Phase_3(protein_results, ligand_results, input_text, output_dir)
    logger.info(f"Phase 3 Results: {results}")