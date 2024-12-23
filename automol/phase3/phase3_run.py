# AutoMol-v2/automol/phase3/run_Phase_3.py

import os
import sys
# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
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


# Importing the separate modules
from phase3.simulate import run_simulation_pipeline
from phase3.docking_simulation import run_docking_pipeline
from phase3.analyze import run_analysis_pipeline
from phase3.VHT_screening import run_vht_screening
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
    print("Starting PDB to PDBQT conversion")
    obConversion.WriteFile(mol, output_pdbqt)
    print("PDB to PDBQT conversion completed")
    return output_pdbqt


def run_Phase_3(
    protein_results: List[Dict[str, Any]],
    ligand_results: List[Dict[str, Any]],
    output_dir: str,

) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    phase3_results = {"simulation_results": [], "docking_results": [], "analysis_results": []}
    device = "cuda"
    try:
        # Ensure output directories exist
        simulation_subdir = os.path.join(output_dir, "simulations")
        docking_dir = os.path.join(output_dir, "docking")
        analysis_dir = os.path.join(output_dir, "analysis")
        
        for i, directory in enumerate([simulation_subdir, docking_dir, analysis_dir]):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created/Verified directory: {directory}")

        # VHT Screening
        logger.info("Starting VHT Screening...")
        print("Starting VHT Screening")
        try:
            smiles_dir = os.path.join(output_dir, "ligands")
            vht_results_dir = os.path.join(output_dir, "vht_results")
            os.makedirs(smiles_dir, exist_ok=True)
            
            # Write SMILES to files
            for i, ligand in enumerate(ligand_results):
                with open(os.path.join(smiles_dir, f"ligand_{i}.smi"), 'w') as f:
                    f.write(ligand['smiles'])
            
            vht_results = run_vht_screening(smiles_dir, vht_results_dir)
            phase3_results["vht_results"] = vht_results
            print("VHT Screening completed")
            logger.info("VHT Screening completed.")
        except Exception as e:
            logger.error(f"Error in VHT Screening: {str(e)}")
            phase3_results["vht_results"] = {"error": str(e)}
            print(f"Error in VHT Screening: {str(e)}")

        # Docking Simulation
        logger.info("Starting Docking Simulation...")
        print("Starting Docking Simulation")
        try:
            docking_results = run_docking_pipeline(protein_results, ligand_results, docking_dir)
            phase3_results["docking_results"] = docking_results
            print("Docking Simulation completed")
            logger.info("Docking Simulation completed.")
        except Exception as e:
            logger.error(f"Error in Docking Simulation: {str(e)}")
            phase3_results["docking_results"] = {"error": str(e)}
            print(f"Error in Docking Simulation: {str(e)}")

        # Molecular Dynamics Simulation
        logger.info("Starting Molecular Dynamics Simulation...")
        print("Starting Molecular Dynamics Simulation")
        try:
            simulation_results = run_simulation_pipeline(protein_results, ligand_results, simulation_subdir, device)
            phase3_results["simulation_results"] = simulation_results
            logger.info("Molecular Dynamics Simulation completed.")
            print("Molecular Dynamics Simulation completed")
        except Exception as e:
            logger.error(f"Error in Molecular Dynamics Simulation: {str(e)}")
            phase3_results["simulation_results"] = {"error": str(e)}
            print(f"Error in Molecular Dynamics Simulation: {str(e)}")

        # Analysis
        logger.info("Starting Analysis...")
        print("Starting Analysis")
        try:
            analysis_results = run_analysis_pipeline(simulation_results, protein_results, ligand_results, analysis_dir, device)
            phase3_results["analysis_results"] = analysis_results
            logger.info("Analysis completed.")
            print("Analysis completed")
        except Exception as e:
            logger.error(f"Error in Analysis: {str(e)}")
            phase3_results["analysis_results"] = {"error": str(e)}
            print(f"Error in Analysis: {str(e)}")

        return phase3_results
    
    except Exception as e:
        logger.error(f"Error in Phase 3: {str(e)}", exc_info=True)
        print(f"Error in Phase 3: {str(e)}")
        return {"error": str(e)}
    finally:
        # Clean up resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Phase 3 run completed. Resources cleaned up.")
        print("Phase 3 run completed. Resources cleaned up.")