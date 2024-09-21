# AutoMol-v2/automol/phase3/docking_simulation.py

import os
import logging
from multiprocessing import Pool, cpu_count
from openbabel import openbabel
import pybel  # Note: pybel is part of Open Babel's Python bindings

logger = logging.getLogger(__name__)


def convert_pdb_to_pdbqt(input_pdb, output_pdbqt):
    """
    Convert PDB file to PDBQT format using Open Babel.

    Parameters:
    - input_pdb: Path to the input PDB file.
    - output_pdbqt: Path to the output PDBQT file.

    Returns:
    - None
    """
    try:
        logger.info(f"Converting {input_pdb} to PDBQT format.")
        # Read the molecule using Pybel
        mol = pybel.readfile("pdb", input_pdb).__next__()
        # Write to PDBQT format
        mol.write("pdbqt", output_pdbqt, overwrite=True)
        logger.info(f"Successfully converted to {output_pdbqt}.")
    except Exception as e:
        logger.error(f"Error converting {input_pdb} to PDBQT: {str(e)}", exc_info=True)
        raise


def run_single_docking(receptor_pdbqt, ligand_pdbqt, output_dir, grid_center, grid_size, exhaustiveness=8, n_poses=10):
    """
    Perform docking for a single ligand against a receptor using AutoDock Vina.

    Parameters:
    - receptor_pdbqt: Path to the receptor PDBQT file.
    - ligand_pdbqt: Path to the ligand PDBQT file.
    - output_dir: Directory to store docking results.
    - grid_center: Tuple of (x, y, z) coordinates for the grid center.
    - grid_size: Tuple of (size_x, size_y, size_z) for the grid box.
    - exhaustiveness: Exhaustiveness parameter for Vina.
    - n_poses: Number of poses to generate.

    Returns:
    - Dictionary containing docking results.
    """
    try:
        logger.info(f"Starting docking for ligand: {ligand_pdbqt}")
        vina = Vina(sf_name='vina')
        vina.set_receptor(receptor_pdbqt)
        vina.set_ligand_from_file(ligand_pdbqt)
        vina.compute_vina_maps(center=grid_center, box_size=grid_size)
        vina.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        scores = vina.energies()
        best_score = min(scores) if scores else None
        output_pdbqt = os.path.join(output_dir, f"docked_{os.path.basename(ligand_pdbqt)}.pdbqt")
        vina.write_poses(output_pdbqt, n_poses=1, overwrite=True)
        logger.info(f"Docking completed for ligand: {ligand_pdbqt} with score: {best_score}")
        
        return {
            "ligand": ligand_pdbqt,
            "docked_pose": output_pdbqt,
            "best_score": best_score
        }
    
    except Exception as e:
        logger.error(f"Error during docking for {ligand_pdbqt}: {str(e)}", exc_info=True)
        return {
            "ligand": ligand_pdbqt,
            "docked_pose": None,
            "best_score": None,
            "error": str(e)
        }


def run_docking_pipeline(protein_results, ligand_results, docking_dir):
    """
    Run docking simulations for multiple ligands against multiple receptors.

    Parameters:
    - protein_results: List of protein result dictionaries.
    - ligand_results: List of ligand result dictionaries.
    - docking_dir: Base directory for docking outputs.

    Returns:
    - List of docking result dictionaries.
    """
    logger.info("Starting Docking Pipeline...")

    # Assuming one receptor for simplicity; modify as needed for multiple receptors
    if not protein_results:
        logger.error("No protein results provided for docking.")
        return []

    receptor_pdb = protein_results[0].get("pdb_file")  # Modify if multiple receptors
    if not receptor_pdb:
        logger.error("Receptor PDB file not found in protein results.")
        return []

    ligands_dir = os.path.join(docking_dir, "ligands")
    os.makedirs(ligands_dir, exist_ok=True)

    # Prepare list of ligand PDBQT files
    ligand_pdbqt_files = []
    for ligand in ligand_results:
        pdb_file = ligand.get("pdb_file")
        if pdb_file:
            pdbqt_file = os.path.splitext(pdb_file)[0] + ".pdbqt"
            try:
                convert_pdb_to_pdbqt(pdb_file, pdbqt_file)
                ligand_pdbqt_files.append(pdbqt_file)
            except Exception as e:
                logger.error(f"Skipping ligand {pdb_file} due to conversion error.")
                continue

    if not ligand_pdbqt_files:
        logger.error("No ligand PDBQT files available for docking.")
        return []

    # Define grid parameters based on receptor
    grid_center, grid_size = calculate_grid_parameters(receptor_pdbqt)

    # Ensure receptor is in PDBQT format; convert if necessary
    receptor_pdbqt = os.path.splitext(receptor_pdb)[0] + ".pdbqt"
    if not os.path.exists(receptor_pdbqt):
        try:
            convert_pdb_to_pdbqt(receptor_pdb, receptor_pdbqt)
            logger.info(f"Converted receptor to PDBQT: {receptor_pdbqt}")
        except Exception as e:
            logger.error(f"Failed to convert receptor to PDBQT: {str(e)}")
            return []

    docking_results = []

    def dock_ligand(ligand_pdbqt):
        return run_single_docking(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            output_dir=docking_dir,
            grid_center=grid_center,
            grid_size=grid_size
        )

    try:
        with Pool(processes=min(cpu_count(), len(ligand_pdbqt_files))) as pool:
            docking_results = pool.map(dock_ligand, ligand_pdbqt_files)
    except Exception as e:
        logger.error(f"Error during multiprocessing docking: {str(e)}", exc_info=True)

    logger.info("Docking Pipeline completed.")
    return docking_results


def calculate_grid_parameters(receptor_pdbqt, buffer=10):
    """
    Calculate grid center and size based on the receptor's active site.

    Parameters:
    - receptor_pdbqt: Path to the receptor PDBQT file.
    - buffer: Additional space around the active site in Angstroms.

    Returns:
    - Tuple containing grid_center and grid_size.
    """
    try:
        logger.info(f"Calculating grid parameters for receptor: {receptor_pdbqt}")
        mol = next(pybel.readfile("pdbqt", receptor_pdbqt))

        # Extract atomic coordinates
        coords = [atom.coords for atom in mol.atoms]
        if not coords:
            raise ValueError("No atomic coordinates found in receptor.")

        # Transpose the list to get separate x, y, z lists
        x_coords, y_coords, z_coords = zip(*coords)

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)

        grid_center = (
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            (min_z + max_z) / 2,
        )
        grid_size = (
            (max_x - min_x) + buffer,
            (max_y - min_y) + buffer,
            (max_z - min_z) + buffer,
        )

        logger.info(f"Calculated grid_center: {grid_center}, grid_size: {grid_size}")
        return grid_center, grid_size

    except Exception as e:
        logger.error(f"Error calculating grid parameters: {str(e)}", exc_info=True)
        # Fallback to default grid parameters
        return (0, 0, 0), (20, 20, 20)