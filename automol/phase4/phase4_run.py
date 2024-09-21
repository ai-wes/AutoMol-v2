import os
import json
import logging
import numpy as np
from utils.save_utils import save_json

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

    def simulate_step(self):
        # Simulate metabolism
        if self.metabolites['glucose'] > 0 and self.metabolites['oxygen'] > 0:
            self.metabolites['glucose'] -= 1
            self.metabolites['oxygen'] -= 1
            self.metabolites['atp'] += 2
            self.growth_rate *= 1.01  # Slight growth increase
        else:
            self.growth_rate *= 0.99  # Slight growth decrease

        # Apply random perturbations
        for perturbation, effect in self.perturbations.items():
            if np.random.random() < 0.1:  # 10% chance of perturbation
                self.growth_rate *= effect
                logger.info(f"Applied perturbation: {perturbation}")

        # Basic homeostasis
        self.growth_rate = max(0.1, min(self.growth_rate, 1.0))

        # Replenish resources (simplified environment interaction)
        self.metabolites['glucose'] = min(self.metabolites['glucose'] + 5, 100)
        self.metabolites['oxygen'] = min(self.metabolites['oxygen'] + 5, 100)

        return self.growth_rate, dict(self.metabolites)

def run_digital_twin_simulation(time_steps=100):
    twin = SimpleDigitalTwin()
    growth_rates = []
    metabolite_history = []

    for _ in range(time_steps):
        growth_rate, metabolites = twin.simulate_step()
        growth_rates.append(growth_rate)
        metabolite_history.append(metabolites)

    return growth_rates, metabolite_history

def run_Phase_4():
    """
    Run Phase 4: Digital Twin Simulation and Analysis
    """
    try:
        logger.info("Starting Phase 4: Digital Twin Simulation")

        # Run the digital twin simulation
        growth_rates, metabolite_history = run_digital_twin_simulation()

        # Analyze results
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

        # Save simulation results
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase4")
        os.makedirs(output_dir, exist_ok=True)
        simulation_output_path = os.path.join(output_dir, "simulation_results.json")
        save_json(simulation_results, simulation_output_path)

        logger.info(f"Phase 4 completed successfully. Final growth rate: {final_growth_rate:.4f}")
        return simulation_results

    except Exception as e:
        logger.error(f"Error in Phase 4: {e}")
        raise

if __name__ == "__main__":
    simulation_results = run_Phase_4()
    print(json.dumps(simulation_results, indent=2))