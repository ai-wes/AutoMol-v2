import numpy as np
import pandas as pd
import logging

class StatisticalAnalysis:
    def __init__(self):
        logging.info("Initialized StatisticalAnalysis.")

    def statistical_analysis(self, results: dict) -> dict:
        try:
            logging.info("Performing statistical analysis...")
            for key, value in results.items():
                if isinstance(value, (int, float, np.ndarray, pd.Series, pd.DataFrame)):
                    if isinstance(value, pd.DataFrame):
                        mean_val = value.mean().mean()
                        std_val = value.std().mean()
                        median_val = value.median().median()
                    else:
                        mean_val = np.mean(value)
                        std_val = np.std(value)
                        median_val = np.median(value)
                    results[f'{key}_stats'] = {
                        'mean': mean_val,
                        'std': std_val,
                        'median': median_val
                    }
                    logging.info(f"Statistical analysis for {key}: Mean={mean_val}, Std={std_val}, Median={median_val}")
            return results
        except Exception as e:
            logging.error(f"Error in statistical_analysis: {e}")
            raise
