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
        print("Initialized SimpleDigitalTwin")

    def simulate_step(self):
        print("Simulating step in SimpleDigitalTwin")
        # Simulate metabolism
        if self.metabolites['glucose'] > 0 and self.metabolites['oxygen'] > 0:
            self.metabolites['glucose'] -= 1
            self.metabolites['oxygen'] -= 1
            self.metabolites['atp'] += 2
            self.growth_rate *= 1.01  # Slight growth increase
            print("Metabolism simulated: resources consumed, ATP produced")
            print("Metabolism simulated")
        else:
            self.growth_rate *= 0.99  # Slight growth decrease
            print("Insufficient resources: growth rate decreased")
            print("Insufficient resources")
        # Apply random perturbations
        for perturbation, effect in self.perturbations.items():
            if np.random.random() < 0.1:  # 10% chance of perturbation
                self.growth_rate *= effect
                print(f"Applied perturbation: {perturbation}, new growth rate: {self.growth_rate:.4f}")
                logger.info(f"Applied perturbation: {perturbation}")
                print(f"Applied perturbation: {perturbation}")

        # Basic homeostasis
        self.growth_rate = max(0.1, min(self.growth_rate, 1.0))
        print(f"Growth rate after homeostasis: {self.growth_rate:.4f}")
        print("Homeostasis applied")
        # Replenish resources (simplified environment interaction)
        self.metabolites['glucose'] = min(self.metabolites['glucose'] + 5, 100)
        self.metabolites['oxygen'] = min(self.metabolites['oxygen'] + 5, 100)
        print(f"Resources replenished. Current levels: {self.metabolites}")
        print("Resources replenished")
        return self.growth_rate, dict(self.metabolites)

def run_digital_twin_simulation(time_steps=100):
    print(f"Starting digital twin simulation for {time_steps} time steps")
    print("Starting digital twin simulation")
    twin = SimpleDigitalTwin()
    growth_rates = []
    metabolite_history = []

    for step in range(time_steps):
        print(f"Simulation step {step + 1}/{time_steps}")
        growth_rate, metabolites = twin.simulate_step()
        growth_rates.append(growth_rate)
        metabolite_history.append(metabolites)

    print("Digital twin simulation completed")
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
        print("Initializing Phase 4")

        # Run the digital twin simulation
        print("Running digital twin simulation")
        print("Running digital twin simulation")
        growth_rates, metabolite_history = run_digital_twin_simulation()

        # Analyze results
        print("Analyzing simulation results")
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
        print("Simulation results processed")

        # Save simulation results
        print("Saving simulation results")
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase4")
        os.makedirs(output_dir, exist_ok=True)
        simulation_output_path = os.path.join(output_dir, "simulation_results.json")
        save_json(simulation_results, simulation_output_path)
        print(f"Simulation results saved to: {simulation_output_path}")
        print("Simulation results saved")
        print("Running bio analysis pipeline")
        print("Running bio analysis pipeline")
        run_bio_analysis_pipeline()
        logger.info(f"Phase 4 completed successfully. Final growth rate: {final_growth_rate:.4f}")
        print("Bio analysis pipeline completed")
        
        # Perform final analysis
        final_analysis_results = perform_final_analysis(phase3_results)
        
        # Save final analysis report
        final_report_path = os.path.join(output_dir, "final_analysis_report.json")
        save_json(final_analysis_results, final_report_path)
        logger.info(f"Final analysis report saved at {final_report_path}.")
        print("Final analysis report saved")
        return final_analysis_results

    except Exception as e:
        logger.error(f"Error in Phase 4: {str(e)}", exc_info=True)
        print(f"Error in Phase 4: {str(e)}")
        raise

def perform_final_analysis(phase3_results):
    print("Performing final analysis")
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
    print("Final analysis completed")
    return final_analysis_results


if __name__ == "__main__":
    print("Running Phase 4 module as main")
    print("Running Phase 4 module as main")
    # Load configuration
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        print("Configuration loaded successfully")
        print("Configuration loaded successfully")
    except FileNotFoundError:
        print("Error: config.json file not found")
        print("Error: config.json file not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in config.json")
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
    print("Mock phase3 results prepared")

    print("Running Phase 4")
    print("Starting Phase 4 execution")
    simulation_results = run_Phase_4(mock_phase3_results, config)
    print("Phase 4 completed. Final simulation results:")
    print(json.dumps(simulation_results, indent=2))
    print("Phase 4 completed")