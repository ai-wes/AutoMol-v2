import os
import logging
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def smiles_to_pdbqt(smiles: str, output_dir: str) -> str:
    """
    Convert SMILES to PDBQT format using RDKit and Open Babel.
    """
    try:
        print("Converting SMILES to PDBQT")
        # Generate 3D coordinates using RDKit
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

        # Save as PDB
        pdb_file = os.path.join(output_dir, f"{Chem.MolToInchiKey(mol)[:12]}.pdb")
        Chem.MolToPDBFile(mol, pdb_file)

        # Convert PDB to PDBQT using Open Babel
        pdbqt_file = os.path.splitext(pdb_file)[0] + ".pdbqt"
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "pdbqt")
        
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, pdb_file)
        obConversion.WriteFile(mol, pdbqt_file)

        logger.info(f"Successfully converted SMILES to PDBQT: {pdbqt_file}")
        print(f"Successfully converted SMILES to PDBQT: {pdbqt_file}")
        return pdbqt_file
    except Exception as e:
        logger.error(f"Error converting SMILES to PDBQT: {str(e)}", exc_info=True)
        print(f"Error converting SMILES to PDBQT: {str(e)}")
        raise

def calculate_grid_parameters(receptor_pdbqt: str, buffer: float = 10.0) -> Tuple[List[float], List[float]]:
    """
    Calculate grid center and size based on the receptor's active site.
    """
    try:
        print(f"Calculating grid parameters for receptor: {receptor_pdbqt}")
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
        print(f"Grid center: {center}, Grid size: {size}")
        return center, size
    except Exception as e:
        logger.error(f"Error calculating grid parameters: {e}", exc_info=True)
        print(f"Error calculating grid parameters: {e}")
        raise

def run_single_docking(receptor_pdbqt: str, ligand_pdbqt: str, output_dir: str, config_file: str) -> Dict[str, Any]:
    """
    Run a single docking simulation using LeDock.
    """
    try:
        print(f"Running docking for ligand: {ligand_pdbqt}")
        logger.info(f"Running docking for ligand: {ligand_pdbqt}")
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Construct the command for LeDock
        command = f"LeDock {config_file} {receptor_pdbqt} {ligand_pdbqt} -o {output_dir}"
        logger.info(f"Docking command: {command}")
        print(f"Docking command: {command}")
        
        # TODO: Uncomment the following lines when LeDock is properly installed
        # result = os.system(command)
        # if result != 0:
        #     raise RuntimeError(f"Docking failed with exit code {result}")
        
        # For now, we'll just log that docking would be performed here
        logger.info("Docking would be performed here. LeDock command execution is currently commented out.")
        print("Docking would be performed here. LeDock command execution is currently commented out.")
        
        # Simulated output for testing
        output_file = os.path.join(output_dir, "simulated_docked.pdbqt")
        with open(output_file, 'w') as f:
            f.write("REMARK VINA RESULT:    -7.5      0.000      0.000\n")
        
        best_score = -7.5  # Simulated score
        
        logger.info(f"Simulated docking completed for ligand: {ligand_pdbqt} with score: {best_score}")
        print(f"Simulated docking completed for ligand: {ligand_pdbqt} with score: {best_score}")
        return {
            "ligand": ligand_pdbqt,
            "docked_pose": output_file,
            "best_score": best_score
        }
    except Exception as e:
        logger.error(f"Error during docking: {e}", exc_info=True)
        print(f"Error during docking: {e}")
        return {
            "ligand": ligand_pdbqt,
            "docked_pose": None,
            "best_score": None,
            "error": str(e)
        }

def dock_ligand(ligand: Dict[str, Any], receptor_pdbqt: str, output_dir: str) -> Dict[str, Any]:
    smiles = ligand.get('smiles')
    if not smiles:
        logger.error(f"Ligand is missing 'smiles' key: {ligand}")
        print(f"Ligand is missing 'smiles' key: {ligand}")
        return {
            "ligand": ligand,
            "docked_pose": None,
            "best_score": None,
            "error": "'smiles' key missing in ligand"
        }
    
    try:
        ligand_pdbqt = smiles_to_pdbqt(smiles, output_dir)
    except Exception as e:
        logger.error(f"Failed to convert ligand {smiles} to PDBQT: {e}")
        print(f"Failed to convert ligand {smiles} to PDBQT: {e}")
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
        print(f"Error during docking for ligand {smiles}: {e}")
        return {
            "ligand": ligand,
            "docked_pose": None,
            "best_score": None,
            "error": "Docking failed"
        }

def run_docking_pipeline(protein_results: List[Dict[str, Any]], ligand_results: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    logger.info("Starting Docking Pipeline...")
    print("Starting Docking Pipeline...")

    if not protein_results:
        logger.error("No protein results provided for docking.")
        print("No protein results provided for docking.")
        return []

    receptor_pdbqt = protein_results[0].get('pdbqt_file')
    if not receptor_pdbqt:
        logger.error("No PDBQT file found for receptor.")
        print("No PDBQT file found for receptor.")
        return []

    try:
        grid_center, grid_size = calculate_grid_parameters(receptor_pdbqt)
    except Exception as e:
        logger.error(f"Failed to calculate grid parameters: {e}")
        print(f"Failed to calculate grid parameters: {e}")
        return []

    docking_results = []

    for ligand in ligand_results:
        result = dock_ligand(ligand, receptor_pdbqt, output_dir)
        docking_results.append(result)

    logger.info("Docking Pipeline completed.")
    print("Docking Pipeline completed.")
    return docking_results

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    protein_results = [
        {'pdbqt_file': r'C:\Users\wes\AutoMol-v2\automol\utils\AF-P07237-F1-model_v4.pdbqt'},  # Telomerase enzyme
    ]
    ligand_results = [
        {'smiles': 'CC1=C(C(=O)NC1=O)N2C=C(C3=C(C=NN32)C#N)C4=CC=C(C=C4)C(F)(F)F'},  # TA-65 (telomerase activator)
        {'smiles': 'CC1=CC(=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'},  # Mitochondria-targeted antioxidant
        {'smiles': 'COC1=C(C=C2C(=C1)C(=O)C=C(O2)C3=CC=CC=C3)OC'}  # Autophagy inducer
    ]
    output_dir = 'aging_docking_results'
    
    print("Starting docking pipeline")
    results = run_docking_pipeline(protein_results, ligand_results, output_dir)

    for result in results:
        if result['docked_pose']:
            print(f"Docking successful for {result['ligand']}. Best score: {result['best_score']}")
            print(f"Docking successful for {result['ligand']}. Best score: {result['best_score']}")
        else:
            print(f"Docking failed for {result['ligand']}. Error: {result['error']}")
            print(f"Docking failed for {result['ligand']}. Error: {result['error']}")

if __name__ == "__main__":
    print("Starting main function")
    main()
    print("Main function completed")