import numpy as np
import pandas as pd
import logging

class TelomereLength:
    def __init__(self):
        logging.info("Initialized TelomereLength.")

    def measure_telomere_length(self, qpcr_data: pd.DataFrame, genome_data: dict = None) -> float:
        try:
            logging.info("Measuring telomere length...")
            if genome_data is not None:
                telomere_length = self.run_telomerecat(genome_data)
            else:
                cq_telomere = qpcr_data['telomere_Cq']
                cq_single_copy = qpcr_data['single_copy_Cq']
                delta_cq = cq_telomere - cq_single_copy
                telomere_length = 2 ** (-delta_cq)
            logging.info(f"Telomere Length: {telomere_length}")
            return telomere_length
        except Exception as e:
            logging.error(f"Error in measure_telomere_length: {e}")
            raise

    def run_telomerecat(self, genome_data: dict) -> float:
        try:
            logging.info("Running Telomerecat analysis...")
            telomere_reads = genome_data.get('telomere_reads', [])
            total_reads = genome_data.get('total_reads', 1)
            genome_size = genome_data.get('genome_size', 3e9)  # Default human genome size
            if not telomere_reads:
                raise ValueError("No telomere reads provided.")
            telomere_length = (np.sum(telomere_reads) / np.sum(total_reads)) * genome_size
            logging.info(f"Telomere Length (Telomerecat): {telomere_length}")
            return telomere_length
        except Exception as e:
            logging.error(f"Error in run_telomerecat: {e}")
            raise
