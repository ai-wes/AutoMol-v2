
import os
from vina import Vina
from Bio.PDB import PDBParser, PDBIO

# Paths to the receptor and ligand PDBQT files
RECEPTOR_PATH = "path/to/receptor.pdbqt"
LIGAND_PATH = "path/to/ligand.pdbqt"
OUTPUT_DIR = "./docking_results"

# Define your grid box parameters
GRID_CENTER = (0, 0, 0)  # Center of the grid box (x, y, z)
GRID_SIZE = (20, 20, 20) # Size of the grid box (x, y, z)

# Function to prepare output directory
def prepare_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Function to run docking using AutoDock Vina
def run_docking(receptor_path, ligand_path, output_dir, grid_center, grid_size):
    vina = Vina(sf_name='vina')

    # Load the receptor and ligand
    vina.set_receptor(receptor_path)
    vina.set_ligand_from_file(ligand_path)

    # Set the docking box
    vina.compute_vina_maps(center=grid_center, box_size=grid_size)

    # Run the docking simulation
    vina.dock(exhaustiveness=8, n_poses=10)

    # Score the best pose
    vina_score = vina.score()
    print(f'Best docking score: {vina_score[0]}')

    # Output the results
    output_path = os.path.join(output_dir, 'docked_ligand.pdbqt')
    vina.write_poses(output_path, n_poses=1, overwrite=True)
    print(f'Results saved to {output_path}')

    return vina_score

# Main function to execute docking
def main():
    prepare_output_directory(OUTPUT_DIR)
    docking_score = run_docking(RECEPTOR_PATH, LIGAND_PATH, OUTPUT_DIR, GRID_CENTER, GRID_SIZE)
    print(f'Docking completed. Best score: {docking_score[0]}')

if __name__ == "__main__":
    main()
