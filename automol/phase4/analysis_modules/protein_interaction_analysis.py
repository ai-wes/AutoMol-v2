import re
import numpy as np
import logging
from Bio import PDB
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import json
import glob
from datetime import datetime

class ProteinLigandInteraction:
    BASE_DIR = r"C:\Users\wes\AutoMol-v2"

    def __init__(self):
        logging.info("Initialized ProteinLigandInteraction.")
        self.pdb_parser = PDB.PDBParser(QUIET=True)

    def analyze_protein_ligand_interaction(self, spr_data: dict, protein_index: int = 0) -> float:
        try:
            logging.info("Analyzing protein-ligand interactions...")
            ka = spr_data.get('association_rate')
            kd = spr_data.get('dissociation_rate')
            if ka is None or kd is None:
                raise ValueError("Missing association_rate or dissociation_rate in spr_data")
            binding_affinity = kd / ka
            logging.info(f"Binding Affinity (Kd): {binding_affinity}")

            protein_pdb = self.get_latest_file(f"results/phase3/protein_{protein_index}/simulation", "final.pdb")
            ligand_smiles = self.get_ligand_smiles()

            if protein_pdb and ligand_smiles:
                docking_score = self.perform_docking(protein_pdb, ligand_smiles)
                logging.info(f"Docking Score: {docking_score}")

            return binding_affinity
        except Exception as e:
            logging.error(f"Error in analyze_protein_ligand_interaction: {e}")
            raise

    def get_latest_file(self, directory, file_pattern):
        try:
            search_path = os.path.join(self.BASE_DIR, directory, file_pattern)
            files = glob.glob(search_path)
            if not files:
                raise FileNotFoundError(f"No files matching {search_path} found")
            return max(files, key=os.path.getctime)
        except Exception as e:
            logging.error(f"Error finding latest file: {e}")
            raise

    def get_ligand_smiles(self):
        try:
            phase2b_dir = os.path.join(self.BASE_DIR, "results/phase2b")
            latest_results_file = self.get_latest_file(phase2b_dir, "phase2b_results_*.json")
            
            with open(latest_results_file, 'r') as f:
                phase2b_results = json.load(f)
            
            if not isinstance(phase2b_results, list) or not phase2b_results:
                raise ValueError("Phase 2b results is not a list or is empty")
            
            # Take the first ligand result
            first_ligand = phase2b_results[0]
            if not isinstance(first_ligand, dict):
                raise ValueError("Ligand result is not a dictionary")
            
            smiles = first_ligand.get('smiles')
            if not smiles:
                raise ValueError("SMILES not found in the first ligand result")
            
            logging.info(f"Retrieved SMILES: {smiles}")
            return smiles
        except Exception as e:
            logging.error(f"Error getting ligand SMILES: {e}")
            raise

    def perform_docking(self, protein_pdb: str, ligand_smiles: str) -> float:
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")
            
            # Pre-process the PDB file
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.readlines()
            
            processed_pdb = []
            atom_counter = 0
            residue_counter = 0
            for line in pdb_content:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_counter += 1
                    residue_counter += 1
                    # Replace any non-digit characters in the atom serial number with spaces
                    processed_line = re.sub(r'[^0-9 ]', ' ', line[:11]) + f"{atom_counter:5d}" + line[16:22]
                    # Replace 'A000' residue number with a numeric value
                    if line[22:26] == 'A000':
                        processed_line += f"{residue_counter:4d}" + line[26:]
                    else:
                        processed_line += line[22:]
                    processed_pdb.append(processed_line)
                else:
                    processed_pdb.append(line)
            
            logging.info(f"Processed {atom_counter} atoms in the PDB file")
            
            # Check if 'A000' is still present in the processed PDB
            processed_pdb_content = ''.join(processed_pdb)
            if 'A000' in processed_pdb_content:
                logging.warning("'A000' is still present in the processed PDB file")
                # Log the lines containing 'A000'
                for i, line in enumerate(processed_pdb):
                    if 'A000' in line:
                        logging.warning(f"Line {i+1} contains 'A000': {line.strip()}")
            else:
                logging.info("'A000' has been successfully removed from the PDB file")
            
            # Write the processed PDB to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as temp_pdb:
                temp_pdb.writelines(processed_pdb)
                temp_pdb_path = temp_pdb.name
            
            logging.info(f"Temporary PDB file created at: {temp_pdb_path}")
            
            logging.info("Performing docking simulation...")
            structure = self.pdb_parser.get_structure("protein", temp_pdb_path)
            
            # Log information about the structure
            for model in structure:
                for chain in model:
                    for residue in chain:
                        logging.info(f"Residue: {residue.id}, Resname: {residue.resname}")
                        break  # Just log the first residue for brevity
                    break
                break
            
            # Clean up the temporary file
            import os
            os.unlink(temp_pdb_path)
            logging.info("Temporary PDB file deleted")
            
            # Continue with the rest of the docking logic...
        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")
            
            # Pre-process the PDB file
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.readlines()
            
            processed_pdb = []
            atom_counter = 0
            for line in pdb_content:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_counter += 1
                    # Replace any non-digit characters in the atom serial number with spaces
                    processed_line = re.sub(r'[^0-9 ]', ' ', line[:11]) + f"{atom_counter:5d}" + line[16:21]
                    # Replace 'A000' chain identifier with 'X'
                    if line[21:25] == 'A000':
                        processed_line += 'X' + line[25:]
                    else:
                        processed_line += line[21:]
                    processed_pdb.append(processed_line)
                else:
                    processed_pdb.append(line)
            
            logging.info(f"Processed {atom_counter} atoms in the PDB file")
            
            # Check if 'A000' is still present in the processed PDB
            processed_pdb_content = ''.join(processed_pdb)
            if 'A000' in processed_pdb_content:
                logging.warning("'A000' is still present in the processed PDB file")
                # Log the lines containing 'A000'
                for i, line in enumerate(processed_pdb):
                    if 'A000' in line:
                        logging.warning(f"Line {i+1} contains 'A000': {line.strip()}")
            else:
                logging.info("'A000' has been successfully removed from the PDB file")
            
            # Write the processed PDB to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as temp_pdb:
                temp_pdb.writelines(processed_pdb)
                temp_pdb_path = temp_pdb.name
            
            logging.info(f"Temporary PDB file created at: {temp_pdb_path}")
            
            logging.info("Performing docking simulation...")
            structure = self.pdb_parser.get_structure("protein", temp_pdb_path)
            
            # Log information about the structure
            for model in structure:
                for chain in model:
                    for residue in chain:
                        logging.info(f"Residue: {residue.id}, Resname: {residue.resname}")
                        break  # Just log the first residue for brevity
                    break
                break
            
            # Clean up the temporary file
            import os
            os.unlink(temp_pdb_path)
            logging.info("Temporary PDB file deleted")
            
            # Continue with the rest of the docking logic...
        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")
            
            # Pre-process the PDB file
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.readlines()
            
            processed_pdb = []
            atom_counter = 0
            residue_counter = 0
            for line in pdb_content:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_counter += 1
                    residue_counter += 1
                    # Replace any non-digit characters in the atom serial number with spaces
                    processed_line = re.sub(r'[^0-9 ]', ' ', line[:11]) + f"{atom_counter:5d}" + line[16:22] + f"{residue_counter:4d}" + line[26:]
                    processed_pdb.append(processed_line)
                else:
                    processed_pdb.append(line)
            
            logging.info(f"Processed {atom_counter} atoms in the PDB file")
            
            # Check if 'A000' is still present in the processed PDB
            processed_pdb_content = ''.join(processed_pdb)
            if 'A000' in processed_pdb_content:
                logging.warning("'A000' is still present in the processed PDB file")
                # Log the lines containing 'A000'
                for i, line in enumerate(processed_pdb):
                    if 'A000' in line:
                        logging.warning(f"Line {i+1} contains 'A000': {line.strip()}")
            else:
                logging.info("'A000' has been successfully removed from the PDB file")
            
            # Write the processed PDB to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as temp_pdb:
                temp_pdb.writelines(processed_pdb)
                temp_pdb_path = temp_pdb.name
            
            logging.info(f"Temporary PDB file created at: {temp_pdb_path}")
            
            logging.info("Performing docking simulation...")
            structure = self.pdb_parser.get_structure("protein", temp_pdb_path)
            
            # Log information about the structure
            for model in structure:
                for chain in model:
                    for residue in chain:
                        logging.info(f"Residue: {residue.id}, Resname: {residue.resname}")
                        break  # Just log the first residue for brevity
                    break
                break
            
            # Clean up the temporary file
            import os
            os.unlink(temp_pdb_path)
            logging.info("Temporary PDB file deleted")
            
            # Continue with the rest of the docking logic...
        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")
            
            # Pre-process the PDB file
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.readlines()
            
            processed_pdb = []
            atom_counter = 0
            for line in pdb_content:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_counter += 1
                    # Replace any non-digit characters in the atom serial number with spaces
                    processed_line = re.sub(r'[^0-9 ]', ' ', line[:11]) + f"{atom_counter:5d}" + line[16:]
                    processed_pdb.append(processed_line)
                else:
                    processed_pdb.append(line)
            
            logging.info(f"Processed {atom_counter} atoms in the PDB file")
            
            # Check if 'A000' is still present in the processed PDB
            processed_pdb_content = ''.join(processed_pdb)
            if 'A000' in processed_pdb_content:
                logging.warning("'A000' is still present in the processed PDB file")
                # Log the lines containing 'A000'
                for i, line in enumerate(processed_pdb):
                    if 'A000' in line:
                        logging.warning(f"Line {i+1} contains 'A000': {line.strip()}")
            else:
                logging.info("'A000' has been successfully removed from the PDB file")
            
            # Write the processed PDB to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as temp_pdb:
                temp_pdb.writelines(processed_pdb)
                temp_pdb_path = temp_pdb.name
            
            logging.info(f"Temporary PDB file created at: {temp_pdb_path}")
            
            logging.info("Performing docking simulation...")
            structure = self.pdb_parser.get_structure("protein", temp_pdb_path)
            
            # Log information about the structure
            for model in structure:
                for chain in model:
                    for residue in chain:
                        logging.info(f"Residue: {residue.id}, Resname: {residue.resname}")
                        break  # Just log the first residue for brevity
                    break
                break
            
            # Clean up the temporary file
            import os
            os.unlink(temp_pdb_path)
            logging.info("Temporary PDB file deleted")
            
            # Continue with the rest of the docking logic...
        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")
            
            # Pre-process the PDB file
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.readlines()
            
            processed_pdb = []
            atom_counter = 0
            for line in pdb_content:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_counter += 1
                    processed_line = f"{line[:6]}{atom_counter:5d}{line[11:]}"
                    processed_pdb.append(processed_line)
                else:
                    processed_pdb.append(line)
            
            logging.info(f"Processed {atom_counter} atoms in the PDB file")
            
            # Check if 'A000' is still present in the processed PDB
            processed_pdb_content = ''.join(processed_pdb)
            if 'A000' in processed_pdb_content:
                logging.warning("'A000' is still present in the processed PDB file")
            else:
                logging.info("'A000' has been successfully removed from the PDB file")
            
            # Write the processed PDB to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as temp_pdb:
                temp_pdb.writelines(processed_pdb)
                temp_pdb_path = temp_pdb.name
            
            logging.info(f"Temporary PDB file created at: {temp_pdb_path}")
            
            logging.info("Performing docking simulation...")
            structure = self.pdb_parser.get_structure("protein", temp_pdb_path)
            
            # Log information about the structure
            for model in structure:
                for chain in model:
                    for residue in chain:
                        logging.info(f"Residue: {residue.id}, Resname: {residue.resname}")
                        break  # Just log the first residue for brevity
                    break
                break
            
            # Clean up the temporary file
            import os
            os.unlink(temp_pdb_path)
            logging.info("Temporary PDB file deleted")
            
            # Continue with the rest of the docking logic...
        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")
            
            # Pre-process the PDB file
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.readlines()
            
            processed_pdb = []
            atom_counter = 0
            for line in pdb_content:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_counter += 1
                    processed_line = f"{line[:6]}{atom_counter:5d}{line[11:]}"
                    processed_pdb.append(processed_line)
                else:
                    processed_pdb.append(line)
            
            # Write the processed PDB to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as temp_pdb:
                temp_pdb.writelines(processed_pdb)
                temp_pdb_path = temp_pdb.name
            
            logging.info("Performing docking simulation...")
            structure = self.pdb_parser.get_structure("protein", temp_pdb_path)
            
            # Log information about the structure
            for model in structure:
                for chain in model:
                    for residue in chain:
                        logging.info(f"Residue: {residue.id}, Resname: {residue.resname}")
                        break  # Just log the first residue for brevity
                    break
                break
            
            # Clean up the temporary file
            import os
            os.unlink(temp_pdb_path)
            
            # Continue with the rest of the docking logic...
        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")
            
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.readlines()
            
            atom_lines = [line for line in pdb_content if line.startswith('ATOM')]
            logging.info(f"First 10 ATOM lines:\n{''.join(atom_lines[:10])}")
            logging.info(f"Last 10 ATOM lines:\n{''.join(atom_lines[-10:])}")
            
            if 'A000' in ''.join(pdb_content):
                logging.warning("'A000' found in PDB file!")
                for i, line in enumerate(pdb_content):
                    if 'A000' in line:
                        logging.warning(f"Line {i+1} contains 'A000': {line.strip()}")
            
            logging.info("Performing docking simulation...")
            structure = self.pdb_parser.get_structure("protein", protein_pdb)
            
            # Log information about the structure
            for model in structure:
                for chain in model:
                    for residue in chain:
                        logging.info(f"Residue: {residue.id}, Resname: {residue.resname}")
                        break  # Just log the first residue for brevity
                    break
                break
            
            # Continue with the rest of the docking logic...
        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")
            
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.readlines()
            
            atom_lines = [line for line in pdb_content if line.startswith('ATOM')]
            logging.info(f"First 10 ATOM lines:\n{''.join(atom_lines[:10])}")
            
            logging.info("Performing docking simulation...")
            structure = self.pdb_parser.get_structure("protein", protein_pdb)
            
            # Log information about the structure
            for model in structure:
                for chain in model:
                    for residue in chain:
                        logging.info(f"Residue: {residue.id}, Resname: {residue.resname}")
                        break  # Just log the first residue for brevity
                    break
                break
            
            # Continue with the rest of the docking logic...
        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")
            
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.read()
            logging.info(f"PDB file content (first 500 chars): {pdb_content[:500]}")
            
            logging.info("Performing docking simulation...")
            structure = self.pdb_parser.get_structure("protein", protein_pdb)
            
            # Log information about the structure
            for model in structure:
                for chain in model:
                    for residue in chain:
                        logging.info(f"Residue: {residue.id}, Resname: {residue.resname}")
                        break  # Just log the first residue for brevity
                    break
                break
            
            # Continue with the rest of the docking logic...
        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise
        logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
        logging.info(f"ligand_smiles: {ligand_smiles}")
        try:
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.read()
            logging.info(f"PDB file content (first 500 chars): {pdb_content[:500]}")
        except Exception as e:
            logging.error(f"Error reading PDB file: {e}")
        
        try:
            logging.info("Performing docking simulation...")
            if not os.path.exists(protein_pdb):
                raise FileNotFoundError(f"PDB file not found: {protein_pdb}")
            
            # Read protein structure
            structure = self.pdb_parser.get_structure("protein", protein_pdb)
            
            # Process ligand
            ligand = Chem.MolFromSmiles(ligand_smiles)
            if ligand is None:
                raise ValueError(f"Invalid SMILES string: {ligand_smiles}")
            
            ligand = Chem.AddHs(ligand)
            AllChem.EmbedMolecule(ligand)
            AllChem.UFFOptimizeMolecule(ligand)
            
            # Simulated docking score (replace with actual docking logic)
            docking_score = np.random.uniform(-10, 0)
            logging.info(f"Simulated Docking Score: {docking_score}")
            return docking_score
        except FileNotFoundError as e:
            logging.error(f"PDB file error: {e}")
            raise
        except ValueError as e:
            logging.error(f"SMILES string error: {e}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            raise