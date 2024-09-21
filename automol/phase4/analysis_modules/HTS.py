import pandas as pd
from scipy import stats
import logging

class HighThroughputScreening:
    def __init__(self):
        logging.info("Initialized HighThroughputScreening.")

    def perform_high_throughput_screening(self, screening_data: pd.Series) -> pd.Series:
        try:
            logging.info("Conducting high-throughput screening...")
            z_scores = stats.zscore(screening_data)
            hits = screening_data[(z_scores > 2) | (z_scores < -2)]
            logging.info(f"Identified {len(hits)} screening hits.")
            return hits
        except Exception as e:
            logging.error(f"Error in perform_high_throughput_screening: {e}")
            raise
