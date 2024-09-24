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
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        logging.info("Initialized ProteinLigandInteraction.")
        self.pdb_parser = PDB.PDBParser(QUIET=True)

    def analyze_protein_ligand_interaction(self, spr_data: dict, protein_pdb: str, ligand_smiles: str) -> float:
        try:
            logging.info("Analyzing protein-ligand interactions...")
            ka = spr_data.get('association_rate')
            kd = spr_data.get('dissociation_rate')
            if ka is None or kd is None:
                raise ValueError("Missing association_rate or dissociation_rate in spr_data")
            binding_affinity = kd / ka
            logging.info(f"Binding Affinity (Kd): {binding_affinity}")

            if protein_pdb and ligand_smiles:
                docking_score = self.perform_docking(protein_pdb, ligand_smiles)
                logging.info(f"Docking Score: {docking_score}")

            return binding_affinity
        except Exception as e:
            logging.error(f"Error in analyze_protein_ligand_interaction: {e}")
            raise

    def get_latest_file(self, directory: str, file_pattern: str) -> str:
        try:
            search_path = os.path.join(self.base_dir, directory, file_pattern)
            files = glob.glob(search_path)
            if not files:
                raise FileNotFoundError(f"No files matching {search_path} found")
            latest_file = max(files, key=os.path.getctime)
            logging.info(f"Latest file found: {latest_file}")
            return latest_file
        except Exception as e:
            logging.error(f"Error finding latest file: {e}")
            raise

    def get_ligand_smiles(self) -> str:
        # Example method to retrieve ligand SMILES, can be modified as needed
        try:
            smiles_file = os.path.join(self.base_dir, 'data', 'ligand.smiles')
            with open(smiles_file, 'r') as file:
                smiles = file.read().strip()
            if not smiles:
                raise ValueError("SMILES string is empty.")
            logging.info(f"Ligand SMILES retrieved: {smiles}")
            return smiles
        except Exception as e:
            logging.error(f"Error retrieving ligand SMILES: {e}")
            raise

    def perform_docking(self, protein_pdb: str, ligand_smiles: str) -> float:
        try:
            logging.info(f"Starting perform_docking with protein_pdb: {protein_pdb}")
            logging.info(f"ligand_smiles: {ligand_smiles}")

            # Pre-process the PDB file
            with open(protein_pdb, 'r') as pdb_file:
                pdb_content = pdb_file.read()
            logging.info(f"PDB file content (first 500 chars): {pdb_content[:500]}")

            # Example docking logic (placeholder)
            # Here, you would integrate with a docking tool like AutoDock Vina
            # For demonstration, we'll simulate a docking score
            docking_score = np.random.uniform(-10, 0)  # Simulated docking score
            logging.info(f"Simulated Docking Score: {docking_score}")

            return docking_score

        except ValueError as ve:
            logging.error(f"ValueError in perform_docking: {ve}")
            logging.error(f"Error details: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error in perform_docking: {e}")
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {str(e)}")
            raise