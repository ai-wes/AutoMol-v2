import logging
import sys
import os
from colorama import Fore, init
import numpy as np
import pandas as pd
import torch
from phase4.analysis_modules.data_loading import DataLoading
from phase4.analysis_modules.protein_interaction_analysis import ProteinLigandInteraction



# Initialize colorama
init(autoreset=True)

# Setup logging
logging.basicConfig(filename='biomolecular_analysis.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import analysis modules (adjust these as needed)
from phase4.analysis_modules.protein_interaction_analysis import ProteinLigandInteraction
from phase4.analysis_modules.bioinformatics_analysis import BioinformaticsAnalysis
from phase4.analysis_modules.data_loading import DataLoading

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"{Fore.GREEN}CUDA cache cleared.")
        logging.info("CUDA cache cleared.")
    else:
        print(f"{Fore.YELLOW}CUDA is not available. Skipping CUDA cache clearing.")
        logging.info("CUDA is not available. Skipping CUDA cache clearing.")

def run_bio_analysis_pipeline():
    try:
        logging.info("Starting Biomolecular Analysis Pipeline.")
        print(f"{Fore.CYAN}Starting Biomolecular Analysis Pipeline.")

        # Initialize analysis classes
        protein_ligand = ProteinLigandInteraction()
        bioinfo = BioinformaticsAnalysis()
        data_loader = DataLoading()

        results = {}

        # Load PDB file (1ERE - ERα LBD with estradiol)
        protein_pdb = data_loader.load_protein_pdb('data/1ERE.pdb')
        print(f"{Fore.BLUE}Protein PDB loaded. Number of atoms: {len(protein_pdb)}")
        clear_cuda_cache()

        # Load or generate ligand data (estradiol SMILES)
        ligand_smiles = "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O"  # Estradiol SMILES
        print(f"{Fore.BLUE}Ligand SMILES loaded: {ligand_smiles}")

        # Protein-Ligand Interaction Analysis
        spr_data = {'association_rate': 3.14e5, 'dissociation_rate': 2.85e-3}  # Example values for estradiol-ERα
        results['binding_affinity'] = protein_ligand.analyze_protein_ligand_interaction(
            spr_data, protein_pdb, ligand_smiles
        )
        print(f"{Fore.GREEN}Binding affinity: {results['binding_affinity']}")
        clear_cuda_cache()

        # Bioinformatics Analysis (e.g., analyzing protein structure)
        results['structure_analysis'] = bioinfo.analyze_protein_structure(protein_pdb)
        print(f"{Fore.GREEN}Structure analysis completed.")
        clear_cuda_cache()

        logging.info("Biomolecular Analysis Pipeline completed successfully.")
        print(f"{Fore.CYAN}Biomolecular Analysis Pipeline completed successfully.")
        print(f"{Fore.WHITE}Final results:")
        for key, value in results.items():
            print(f"{Fore.WHITE}{key}: {value}")

    except Exception as e:
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        logging.critical(f"Pipeline execution failed: {e}")
        print(f"{Fore.RED}Pipeline execution failed: {e}")

if __name__ == "__main__":
    run_bio_analysis_pipeline()