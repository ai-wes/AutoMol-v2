import os
import json
import logging
from utils.save_utils import save_json, load_json
from phase4.BiomolecularAnalysisPipeline import BiomolecularAnalysisPipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


biomolecular_analysis_pipeline = BiomolecularAnalysisPipeline()

def run_Phase_4(phase3_results):
    """
    Run Phase 4: Virtual Lab Simulation and Automation based on Phase 3 results.
    """
    try:
        logger.info("Starting Phase 4: Virtual Lab Simulation and Automation")

        # Example: Load phase3_results from JSON if it's a path
        if isinstance(phase3_results, str):
            phase3_data = load_json(phase3_results)
        elif isinstance(phase3_results, dict):
            phase3_data = phase3_results
        else:
            raise ValueError("phase3_results must be a dict or a path to a JSON file.")

        # Extract necessary data from phase3_data
        digital_twin_growth_rate = phase3_data.get("digital_twin_growth_rate", None)
        if digital_twin_growth_rate is None:
            logger.error("Digital twin growth rate not found in Phase 3 results.")
            raise ValueError("Missing digital twin growth rate.")

        # Perform virtual lab simulations based on growth rate
        # Placeholder for simulation logic
        simulation_results = {
            "growth_rate": digital_twin_growth_rate,
            "simulation_status": "Success",
            "details": "Simulated environment based on digital twin."
        }

        # Save simulation results
        simulation_output_path = os.path.join("phase4", "simulation_results.json")
        save_json(simulation_results, simulation_output_path)

        logger.info("Phase 4 completed successfully.")
        return simulation_results

    except Exception as e:
        logger.error(f"Error in Phase 4: {e}")
        raise

if __name__ == "__main__":
    def test_run():
        phase3_results_path = "path/to/phase3/phase3_results.json"
        simulation_results = run_Phase_4(phase3_results_path)
        print(simulation_results)

    test_run()