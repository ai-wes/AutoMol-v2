import os   
import sys
# Add the parent directory to the Python path




current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import logging
import json
import numpy as np
from utils.save_utils import save_json
from phase4.analysis_modules.setup_logger import setup_logger
from phase4.analysis_modules.base_bio_analysis import run_bio_analysis_pipeline
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("Starting Phase 4 module")

def setup_logger(log_file_path):
    """
    Set up and configure a logger.

    Args:
    log_file_path (str): Path to the log file.

    Returns:
    logging.Logger: Configured logger object.
    """
    # Create the directory for the log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger setup complete.")
    return logger


class SimpleDigitalTwin:
    def __init__(self, initial_growth_rate=0.5):
        self.growth_rate = initial_growth_rate
        self.metabolites = {
            'glucose': 100,
            'oxygen': 100,
            'atp': 50
        }
        self.perturbations = {
            'nutrient_depletion': 0.95,
            'oxidative_stress': 0.97,
            'metabolic_boost': 1.03
        }
        print(f"Initialized SimpleDigitalTwin with growth rate: {self.growth_rate}")

    def simulate_step(self):
        print("Simulating step in SimpleDigitalTwin")
        # Simulate metabolism
        if self.metabolites['glucose'] > 0 and self.metabolites['oxygen'] > 0:
            self.metabolites['glucose'] -= 1
            self.metabolites['oxygen'] -= 1
            self.metabolites['atp'] += 2
            self.growth_rate *= 1.01  # Slight growth increase
            print("Metabolism simulated: resources consumed, ATP produced")
        else:
            self.growth_rate *= 0.99  # Slight growth decrease
            print("Insufficient resources: growth rate decreased")

        # Apply random perturbations
        for perturbation, effect in self.perturbations.items():
            if np.random.random() < 0.1:  # 10% chance of perturbation
                self.growth_rate *= effect
                print(f"Applied perturbation: {perturbation}, new growth rate: {self.growth_rate:.4f}")
                logger.info(f"Applied perturbation: {perturbation}")

        # Basic homeostasis
        self.growth_rate = max(0.1, min(self.growth_rate, 1.0))
        print(f"Growth rate after homeostasis: {self.growth_rate:.4f}")

        # Replenish resources (simplified environment interaction)
        self.metabolites['glucose'] = min(self.metabolites['glucose'] + 5, 100)
        self.metabolites['oxygen'] = min(self.metabolites['oxygen'] + 5, 100)
        print(f"Resources replenished. Current levels: {self.metabolites}")

        return self.growth_rate, dict(self.metabolites)

def run_digital_twin_simulation(time_steps=100):
    print(f"Starting digital twin simulation for {time_steps} time steps")
    twin = SimpleDigitalTwin()
    growth_rates = []
    metabolite_history = []

    for step in range(time_steps):
        print(f"Simulation step {step + 1}/{time_steps}")
        growth_rate, metabolites = twin.simulate_step()
        growth_rates.append(growth_rate)
        metabolite_history.append(metabolites)

    print("Digital twin simulation completed")
    return growth_rates, metabolite_history

def run_Phase_4(phase3_results, config):
    """
    Run Phase 4: Digital Twin Simulation and Analysis
    """
    try:
        print("Initializing Phase 4: Digital Twin Simulation and Analysis")
        logger = setup_logger(config['output_paths']['log_file'])
        logger.info("Starting Phase 4: Digital Twin Simulation")

        # Run the digital twin simulation
        print("Running digital twin simulation")
        growth_rates, metabolite_history = run_digital_twin_simulation()

        # Analyze results
        print("Analyzing simulation results")
        final_growth_rate = growth_rates[-1]
        avg_growth_rate = np.mean(growth_rates)
        min_growth_rate = min(growth_rates)
        max_growth_rate = max(growth_rates)

        simulation_results = {
            "final_growth_rate": final_growth_rate,
            "average_growth_rate": avg_growth_rate,
            "min_growth_rate": min_growth_rate,
            "max_growth_rate": max_growth_rate,
            "final_metabolite_levels": metabolite_history[-1],
            "simulation_status": "Success",
            "details": "Digital twin simulation completed successfully."
        }


        print("Simulation results:")
        print(json.dumps(simulation_results, indent=2))

        # Save simulation results
        print("Saving simulation results")
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase4")
        os.makedirs(output_dir, exist_ok=True)
        simulation_output_path = os.path.join(output_dir, "simulation_results.json")
        save_json(simulation_results, simulation_output_path)
        print(f"Simulation results saved to: {simulation_output_path}")

        print("Running bio analysis pipeline")
        run_bio_analysis_pipeline()
        logger.info(f"Phase 4 completed successfully. Final growth rate: {final_growth_rate:.4f}")
        print(f"Phase 4 completed successfully. Final growth rate: {final_growth_rate:.4f}")
        
        # Perform final analysis
        final_analysis_results = perform_final_analysis(phase3_results)
        
        # Save final analysis report
        final_report_path = os.path.join(output_dir, "final_analysis_report.json")
        save_json(final_analysis_results, final_report_path)
        logger.info(f"Final analysis report saved at {final_report_path}.")

        return final_analysis_results

    except Exception as e:
        logger.error(f"Error in Phase 4: {str(e)}", exc_info=True)
        raise

def perform_final_analysis(phase3_results):
    print("Performing final analysis")
    # Example analysis: Calculate the average score from phase3 results
    if phase3_results and 'simulation_results' in phase3_results and phase3_results['simulation_results']:
        average_score = sum(result['score'] for result in phase3_results['simulation_results']) / len(phase3_results['simulation_results'])
    else:
        average_score = 0

    final_analysis_results = {
        'average_score': average_score
    }
    print("Final analysis results:")
    print(json.dumps(final_analysis_results, indent=2))
    return final_analysis_results


if __name__ == "__main__":
    print("Running Phase 4 module as main")
    # Load configuration
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        print("Configuration loaded successfully")
    except FileNotFoundError:
        print("Error: config.json file not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in config.json")
        sys.exit(1)

    # Mock phase3_results for testing
    mock_phase3_results = {
        'simulation_results': [{'score': 0.7}],
        'docking_results': [{'score': 0.6}]
    }
    print("Using mock phase3 results for testing")
    print("Mock phase3 results:")
    print(json.dumps(mock_phase3_results, indent=2))

    print("Running Phase 4")
    simulation_results = run_Phase_4(mock_phase3_results, config)
    print("Phase 4 completed. Final simulation results:")
    print(json.dumps(simulation_results, indent=2))