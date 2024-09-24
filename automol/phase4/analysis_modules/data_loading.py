import os
from Bio import SeqIO
import pandas as pd
import numpy as np
import logging

class DataLoading:
    def __init__(self):
        logging.info("Initialized DataLoading.")

    def load_protein_sequence(self, fasta_file: str) -> str:
        try:
            logging.info(f"Loading protein sequence from {fasta_file}...")
            with open(fasta_file, 'r') as file:
                records = list(SeqIO.parse(file, 'fasta'))
                if not records:
                    raise ValueError("No records found in FASTA file.")
                sequence = str(records[0].seq)
                logging.info("Protein sequence loaded successfully.")
                return sequence
        except Exception as e:
            logging.error(f"Error in load_protein_sequence: {e}")
            raise

    def load_ligand_smiles(self, smiles_file: str) -> str:
        try:
            logging.info(f"Loading ligand SMILES from {smiles_file}...")
            with open(smiles_file, 'r') as file:
                smiles = file.read().strip()
                if not smiles:
                    raise ValueError("SMILES string is empty.")
                logging.info("Ligand SMILES loaded successfully.")
                return smiles
        except Exception as e:
            logging.error(f"Error in load_ligand_smiles: {e}")
            raise

    def load_protein_pdb(self, pdb_file: str) -> str:
        try:
            logging.info(f"Loading protein PDB from {pdb_file}...")
            if not os.path.exists(pdb_file):
                raise FileNotFoundError(f"PDB file {pdb_file} does not exist.")
            logging.info("Protein PDB loaded successfully.")
            return pdb_file
        except Exception as e:
            logging.error(f"Error in load_protein_pdb: {e}")
            raise

    def load_trajectory_file(self, traj_file: str) -> np.ndarray:
        try:
            logging.info(f"Loading trajectory file from {traj_file}...")
            if not os.path.exists(traj_file):
                raise FileNotFoundError(f"Trajectory file {traj_file} does not exist.")
            # Replace with actual loading logic
            trajectory_data = np.random.rand(1000, 3)  # Simulated data
            logging.info("Trajectory file loaded successfully.")
            return trajectory_data
        except Exception as e:
            logging.error(f"Error in load_trajectory_file: {e}")
            raise