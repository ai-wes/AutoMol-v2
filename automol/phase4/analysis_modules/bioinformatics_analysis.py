import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

class BioinformaticsAnalysis:
    def __init__(self):
        logging.info("Initialized BioinformaticsAnalysis.")

    def run_bioinformatics_analysis(self, sequencing_data: pd.DataFrame) -> np.ndarray:
        try:
            logging.info("Running bioinformatics analysis...")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(sequencing_data)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            logging.info("PCA analysis completed.")
            return pca_result
        except Exception as e:
            logging.error(f"Error in run_bioinformatics_analysis: {e}")
            raise
