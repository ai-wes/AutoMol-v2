from automol.emit_progress import emit_progress
import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_pdb_to_pdbqt(pdb_file_path, output_dir):
    """
    Convert a PDB file to PDBQT format.
    
    Args:
        pdb_file_path (str): Path to the input PDB file.
        output_dir (str): Directory to save the output PDBQT file.
    
    Returns:
        Path: Path to the output PDBQT file if successful, None otherwise.
    """
    try:
        # Ensure the input file exists
        pdb_path = Path(pdb_file_path)
        if not pdb_path.is_file():
            raise FileNotFoundError(f"Input PDB file not found: {pdb_file_path}")

        # Ensure the output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Define the output file path
        output_file_path = output_path / f"{pdb_path.stem}.pdbqt"

        # Run the obabel command to convert PDB to PDBQT
        command = ["obabel", str(pdb_path), "-O", str(output_file_path), "-p", "7.4"]
        
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        logging.info(f"Successfully converted {pdb_file_path} to {output_file_path}")
        return output_file_path

    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running obabel command: {e}")
        logging.error(f"Command output: {e.output}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return None

# Example usage
if __name__ == "__main__":
    pdb_file_path = r"C:\Users\wes\AutoMol-v2\automol\utils\AF-P07237-F1-model_v4.pdb"
    output_dir = r"C:\Users\wes\AutoMol-v2\automol\utils"
    result = convert_pdb_to_pdbqt(pdb_file_path, output_dir)
    
    if result:
        emit_progress("conversion_success", {"message": f"Conversion successful. Output file: {result}"})
        print(f"Conversion successful. Output file: {result}")
    else:
        emit_progress("conversion_failure", {"message": "Conversion failed. Check the logs for more information."})
        print("Conversion failed. Check the logs for more information.")