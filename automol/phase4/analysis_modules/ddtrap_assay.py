import numpy as np
import pandas as pd
import logging

class DdTrapAssay:
    def __init__(self):
        logging.info("Initialized DdTrapAssay.")

    def run_ddtrap_assay(self, ddpcr_data: pd.DataFrame) -> float:
        try:
            logging.info("Running ddTRAP assay...")
            positive_droplets = np.sum(ddpcr_data['amplitude'] > ddpcr_data['threshold'])
            total_droplets = len(ddpcr_data)
            telomerase_activity = positive_droplets / total_droplets
            logging.info(f"Telomerase Activity: {telomerase_activity}")
            return telomerase_activity
        except Exception as e:
            logging.error(f"Error in run_ddtrap_assay: {e}")
            raise
