import pandas as pd
import numpy as np
import logging

class GeneExpression:
    def __init__(self):
        logging.info("Initialized GeneExpression.")

    def analyze_gene_expression(self, rna_seq_data: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Analyzing gene expression...")
            counts = rna_seq_data.pivot_table(index='gene', columns='sample', values='counts').fillna(0)
            normalized_counts = counts.div(counts.sum(axis=0), axis=1) * 1e6  # TPM normalization
            telomere_genes = ['TERT', 'TERC', 'DKC1', 'TINF2']
            telomere_gene_expression = normalized_counts.loc[telomere_genes]
            logging.info("Gene expression analysis completed.")
            return telomere_gene_expression
        except KeyError as e:
            logging.error(f"Gene not found in RNA-seq data: {e}")
            raise
        except Exception as e:
            logging.error(f"Error in analyze_gene_expression: {e}")
            raise
