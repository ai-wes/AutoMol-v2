# AutoMol-v2/automol/phase3/docking_simulation.py

import os
import logging
from multiprocessing import Pool, cpu_count
from openbabel import openbabel
from rdkit import Chem
from typing import List, Dict, Any
from typing import List, Tuple
logger = logging.getLogger(__name__)

def convert_pdb_to_pdbqt(input_pdb: str, output_pdbqt: str) -> None:
    """
    Convert PDB file to PDBQT format using Open Babel.
    """
    try:
        logger.info(f"Converting {input_pdb} to PDBQT format.")
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "pdbqt")
        
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, input_pdb)
        obConversion.WriteFile(mol, output_pdbqt)
        
        logger.info(f"Successfully converted to {output_pdbqt}.")
    except Exception as e:
        logger.error(f"Error converting {input_pdb} to PDBQT: {str(e)}", exc_info=True)
        raise

def calculate_grid_parameters(receptor_pdbqt: str, buffer: float = 10.0) -> Tuple[List[float], List[float]]:
    """
    Calculate grid center and size based on the receptor's active site.
    """
    try:
        logger.info(f"Calculating grid parameters for receptor: {receptor_pdbqt}")
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdbqt", "pdbqt")
        
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, receptor_pdbqt)

        # Extract atomic coordinates
        coords = [(atom.GetX(), atom.GetY(), atom.GetZ()) for atom in openbabel.OBMolAtomIter(mol)]
        if not coords:
            raise ValueError("No atomic coordinates found in receptor.")

        # Calculate center
        x_coords, y_coords, z_coords = zip(*coords)
        center = [sum(x_coords) / len(x_coords),
                  sum(y_coords) / len(y_coords),
                  sum(z_coords) / len(z_coords)]
        
        # Define grid size with buffer
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)

        size = [
            (max_x - min_x) + 2 * buffer,
            (max_y - min_y) + 2 * buffer,
            (max_z - min_z) + 2 * buffer
        ]

        logger.info(f"Grid center: {center}, Grid size: {size}")
        return center, size
    except Exception as e:
        logger.error(f"Error calculating grid parameters: {e}", exc_info=True)
        raise

def run_single_docking(receptor_pdbqt: str, ligand_pdbqt: str, output_dir: str, config_file: str) -> Dict[str, Any]:
    """
    Run a single docking simulation using LeDock.
    """
    try:
        logger.info(f"Running docking for ligand: {ligand_pdbqt}")
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Construct the command for LeDock
        command = f"LeDock {config_file} {receptor_pdbqt} {ligand_pdbqt} -o {output_dir}"
        logger.info(f"Docking command: {command}")
        
        # Execute the command
        result = os.system(command)
        if result != 0:
            raise RuntimeError(f"Docking failed with exit code {result}")
        
        # Parse the docking results
        output_file = os.path.join(output_dir, "docked.pdbqt")
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Docking output file not found: {output_file}")
        
        # Extract the best score from the output file
        best_score = None
        with open(output_file, 'r') as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    best_score = float(line.split()[3])
                    break
        
        if best_score is None:
            raise ValueError("Best score not found in docking output")
        
        logger.info(f"Docking completed successfully for ligand: {ligand_pdbqt} with score: {best_score}")
        return {
            "ligand": ligand_pdbqt,
            "docked_pose": output_file,
            "best_score": best_score
        }
    except Exception as e:
        logger.error(f"Error during docking: {e}", exc_info=True)
        return {
            "ligand": ligand_pdbqt,
            "docked_pose": None,
            "best_score": None,
            "error": str(e)
        }

def run_docking_pipeline(protein_results: List[Dict[str, Any]], ligand_results: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    logger.info("Starting Docking Pipeline...")

    # Ensure protein_results is not empty
    if not protein_results:
        logger.error("No protein results provided for docking.")
        return []

    # Use the first protein result for docking
    receptor_pdbqt = protein_results[0].get('pdbqt_file')
    if not receptor_pdbqt:
        logger.error("No PDBQT file found for receptor.")
        return []

    # Calculate grid parameters
    try:
        grid_center, grid_size = calculate_grid_parameters(receptor_pdbqt)
    except Exception as e:
        logger.error(f"Failed to calculate grid parameters: {e}")
        return []

    docking_results = []

    def dock_ligand(ligand: Dict[str, Any]) -> Dict[str, Any]:
        smiles = ligand.get('smiles')
        if not smiles:
            logger.error(f"Ligand is missing 'smiles' key: {ligand}")
            return {
                "ligand": ligand,
                "docked_pose": None,
                "best_score": None,
                "error": "'smiles' key missing in ligand"
            }
        
        # Ensure SMILES is valid
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES string: {smiles}")
            return {
                "ligand": ligand,
                "docked_pose": None,
                "best_score": None,
                "error": "Invalid SMILES string"
            }

        # Convert SMILES to PDBQT if not already converted
        ligand_pdbqt = os.path.splitext(smiles)[0] + ".pdbqt"
        try:
            convert_pdb_to_pdbqt(smiles, ligand_pdbqt)
            logger.info(f"Converted ligand to PDBQT: {ligand_pdbqt}")
        except Exception as e:
            logger.error(f"Failed to convert ligand {smiles} to PDBQT: {e}")
            return {
                "ligand": ligand,
                "docked_pose": None,
                "best_score": None,
                "error": "PDBQT conversion failed"
            }

        # Run docking
        try:
            docking_result = run_single_docking(
                receptor_pdbqt=receptor_pdbqt,
                ligand_pdbqt=ligand_pdbqt,
                output_dir=output_dir,
                config_file='path/to/config.txt'  # Update this path accordingly
            )
            return docking_result
        except Exception as e:
            logger.error(f"Error during docking for ligand {smiles}: {e}")
            return {
                "ligand": ligand,
                "docked_pose": None,
                "best_score": None,
                "error": "Docking failed"
            }

    # Utilize multiprocessing for docking
    try:
        pool_size = min(cpu_count(), len(ligand_results))
        with Pool(processes=pool_size) as pool:
            docking_results = pool.map(dock_ligand, ligand_results)
    except Exception as e:
        logger.error(f"Error during multiprocessing docking: {str(e)}", exc_info=True)

    logger.info("Docking Pipeline completed.")
    return docking_results