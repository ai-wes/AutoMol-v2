
import logging
import sys

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

# Create file handler and set level to debug
fh = logging.FileHandler('biomolecular_analysis.log', mode='w')
fh.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to handlers
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

import logging
import sys
import os

from colorama import Fore, Back, Style, init
# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# Import analysis modules
from phase4.analysis_modules.ddtrap_assay import DdTrapAssay
from phase4.analysis_modules.telomere_length_analysis import TelomereLength
from phase4.analysis_modules.gene_expression_analysis import GeneExpression
from phase4.analysis_modules.protein_interaction_analysis import ProteinLigandInteraction
from phase4.analysis_modules.bioinformatics_analysis import BioinformaticsAnalysis
from phase4.analysis_modules.flow_cytometry import FlowCytometryAnalysis
from phase4.analysis_modules.telomerase_localization import TelomeraseLocalization
from phase4.analysis_modules.HTS import HighThroughputScreening
from phase4.analysis_modules.statistical_analysis import StatisticalAnalysis
from phase4.analysis_modules.analysis_visualization import Visualization
from phase4.analysis_modules.data_loading import DataLoading

import numpy as np
import pandas as pd
import torch  # Import PyTorch for CUDA management

# Initialize colorama
init(autoreset=True)

def clear_cuda_cache():
    """
    Clears the CUDA cache to preserve memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"{Fore.GREEN}CUDA cache cleared.")
        logging.info("CUDA cache cleared.")
    else:
        print(f"{Fore.YELLOW}CUDA is not available. Skipping CUDA cache clearing.")
        logging.info("CUDA is not available. Skipping CUDA cache clearing.")


def run_bio_analysis_pipeline():
    try:
        # Setup logging
        logging.info("Starting Biomolecular Analysis Pipeline.")
        print(f"{Fore.CYAN}Starting Biomolecular Analysis Pipeline.")

        # Initialize analysis classes
        ddtrap = DdTrapAssay()
        tel_length = TelomereLength()
        gene_expr = GeneExpression()
        protein_ligand = ProteinLigandInteraction()
        bioinfo = BioinformaticsAnalysis()
        flow_cyto = FlowCytometryAnalysis()
        tel_loc = TelomeraseLocalization()
        hts = HighThroughputScreening()
        stats_analysis = StatisticalAnalysis()
        viz = Visualization()
        data_loader = DataLoading()

        results = {}

        # Simulate or load data
        ddpcr_data = pd.DataFrame({
            'amplitude': np.random.normal(5000, 200, 10000),
            'threshold': np.repeat(4800, 10000)
        })
        print(f"{Fore.MAGENTA}ddPCR data shape: {ddpcr_data.shape}")

        qpcr_data = pd.DataFrame({
            'telomere_Cq': np.random.normal(20, 1, 100),
            'single_copy_Cq': np.random.normal(25, 1, 100)
        })
        print(f"{Fore.MAGENTA}qPCR data shape: {qpcr_data.shape}")

        rna_seq_data = pd.DataFrame({
            'gene': ['TERT', 'TERC', 'DKC1', 'TINF2'] * 25,
            'sample': np.repeat(['Sample1', 'Sample2'], 50),
            'counts': np.random.poisson(100, 100)
        })
        print(f"{Fore.MAGENTA}RNA-seq data shape: {rna_seq_data.shape}")

        spr_data = {'association_rate': 1e5, 'dissociation_rate': 1e-3}
        print(f"{Fore.MAGENTA}SPR data: {spr_data}")

        sequencing_data = pd.DataFrame(np.random.normal(0, 1, (100, 10)), columns=[f'Gene{i}' for i in range(10)])
        print(f"{Fore.MAGENTA}Sequencing data shape: {sequencing_data.shape}")

        fcs_data = pd.DataFrame(np.random.normal(50, 10, (1000, 3)), columns=['CD34', 'CD38', 'CD90'])
        print(f"{Fore.MAGENTA}Flow cytometry data shape: {fcs_data.shape}")

        image_data = np.random.rand(256, 256)
        print(f"{Fore.MAGENTA}Image data shape: {image_data.shape}")

        screening_data = pd.Series(np.random.normal(0, 1, 10000))
        print(f"{Fore.MAGENTA}Screening data shape: {screening_data.shape}")

        # Run analyses with CUDA cache clearing after each step
        # 1. ddTRAP Assay
        results['telomerase_activity'] = ddtrap.run_ddtrap_assay(ddpcr_data)
        print(f"{Fore.GREEN}Telomerase activity: {results['telomerase_activity']}")
        clear_cuda_cache()

        # 2. Telomere Length Measurement
        results['telomere_length'] = tel_length.measure_telomere_length(qpcr_data)
        print(f"{Fore.GREEN}Telomere length: {results['telomere_length']}")
        clear_cuda_cache()

        # 3. Gene Expression Analysis
        results['gene_expression'] = gene_expr.analyze_gene_expression(rna_seq_data)
        print(f"{Fore.GREEN}Gene expression results: {results['gene_expression']}")
        clear_cuda_cache()

        # 4. Validate Telomerase Activity
        results['predicted_telomerase_activity'] = (
            (results['gene_expression'].loc['TERT'] + results['gene_expression'].loc['TERC']) / 2
        )
        print(f"{Fore.GREEN}Predicted telomerase activity: {results['predicted_telomerase_activity']}")
        clear_cuda_cache()

        # 5. Protein-Ligand Interaction Analysis
        spr_data = {'association_rate': 1e5, 'dissociation_rate': 1e-3}
        results['binding_affinity'] = protein_ligand.analyze_protein_ligand_interaction(
            spr_data, protein_index=0
        )
        print(f"{Fore.GREEN}Binding affinity: {results['binding_affinity']}")
        clear_cuda_cache()


        # 6. Bioinformatics Analysis
        results['pca_result'] = bioinfo.run_bioinformatics_analysis(sequencing_data)
        print(f"{Fore.GREEN}PCA result shape: {results['pca_result'].shape}")
        clear_cuda_cache()

        print("Starting Flow Cytometry Analysis")
# 7. Flow Cytometry Analysis
        results['stem_cell_population'] = flow_cyto.analyze_flow_cytometry(fcs_data)
        print(f"{Fore.GREEN}Stem cell population: {results['stem_cell_population']}")
        clear_cuda_cache()

        print("Starting Telomerase Localization Analysis")
# 8. Telomerase Localization Analysis
        results['telomerase_localization'] = tel_loc.analyze_telomerase_localization(image_data)
        print(f"{Fore.GREEN}Telomerase localization: {results['telomerase_localization']}")
        clear_cuda_cache()

        print("Starting High-Throughput Screening")
# 9. High-Throughput Screening
        results['screening_hits'] = hts.perform_high_throughput_screening(screening_data)
        print(f"{Fore.GREEN}Screening hits: {results['screening_hits']}")
        clear_cuda_cache()

        print("Starting Statistical Analysis")
# 10. Statistical Analysis
        results = stats_analysis.statistical_analysis(results)
        print(f"{Fore.GREEN}Statistical analysis completed. Results updated.")
        clear_cuda_cache()

        print("Starting Visualization")
# 11. Visualization
        viz.visualize_results(results)
        print(f"{Fore.GREEN}Visualization completed.")
        clear_cuda_cache()

        # Load additional inputs
        protein_seq = data_loader.load_protein_sequence('data/protein.fasta')
        print(f"{Fore.BLUE}Protein sequence loaded. Length: {len(protein_seq)}")
        clear_cuda_cache()

        ligand_smiles = data_loader.load_ligand_smiles('data/ligand.smiles')
        print(f"{Fore.BLUE}Ligand SMILES loaded: {ligand_smiles}")
        clear_cuda_cache()

        protein_pdb = data_loader.load_protein_pdb('data/protein.pdb')
        print(f"{Fore.BLUE}Protein PDB loaded. Number of atoms: {len(protein_pdb)}")
        clear_cuda_cache()

        trajectory_data = data_loader.load_trajectory_file('data/trajectory.dcd')
        print(f"{Fore.BLUE}Trajectory data loaded. Number of frames: {len(trajectory_data)}")
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

def clear_cuda_cache():
    """
    Clears the CUDA cache to preserve memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"{Fore.GREEN}CUDA cache cleared.")
        logging.info("CUDA cache cleared.")
    else:
        print(f"{Fore.YELLOW}CUDA is not available. Skipping CUDA cache clearing.")
        logging.info("CUDA is not available. Skipping CUDA cache clearing.")

if __name__ == "__main__":
    run_bio_analysis_pipeline()