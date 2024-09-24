import pandas as pd
from scipy import stats
import logging

class HighThroughputScreening:
    def __init__(self):
        logging.info("Initialized HighThroughputScreening.")

    def perform_high_throughput_screening(self, screening_data: pd.Series) -> pd.Series:
        try:
            logging.info("Performing high-throughput screening analysis...")
            # Example statistical test: z-score normalization
            z_scores = stats.zscore(screening_data)
            screening_hits = pd.Series(z_scores, index=screening_data.index)
            logging.info("High-throughput screening analysis completed.")
            return screening_hits
        except Exception as e:
            logging.error(f"Error in perform_high_throughput_screening: {e}")
            raise