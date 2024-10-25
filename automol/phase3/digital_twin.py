import cobra
from cobra import Solution
from cobra.flux_analysis import (
    flux_variability_analysis,
    single_gene_deletion,
    single_reaction_deletion,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)


class DigitalTwinSimulator:
    def __init__(self, model_path: str):
        """
        Initializes the Digital Twin Simulator with a metabolic model.

        Parameters:
            model_path (str): Path to the metabolic model file in JSON format.
        """
        try:
            self.model = cobra.io.load_json_model(model_path)
            logging.info(f"Model loaded successfully from {model_path}")
            self.original_bounds = {r.id: r.bounds for r in self.model.reactions}
            logging.info("Original bounds set.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.model = None
            raise ValueError(
                "Model could not be loaded. Please check the model path and format."
            )


    def reset_model(self):

        if self.model is None:
            raise ValueError("No model is loaded.")

        for reaction_id, bounds in self.original_bounds.items():
            try:
                reaction = self.model.reactions.get_by_id(reaction_id)
                reaction.bounds = bounds
                logging.debug(f"Reset reaction {reaction_id} to original bounds: {bounds}")
            except KeyError:
                logging.warning(f"Reaction {reaction_id} not found.")
            except Exception as e:
                logging.error(f"Error resetting bounds for reaction {reaction_id}: {e}")
                raise

    def simulate_cellular_environment(
        self,
        drug_effects: Dict[str, float],
        gene_knockouts: Optional[List[str]] = None,
        condition_changes: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, cobra.Solution]:

        if self.model is None:
                raise ValueError("No model is loaded.")

        # Reset the model to its original state
        self.reset_model()
        logging.info("Model reset to original state.")
        print("10Model reset to original state")

        # Apply drug effects
        for reaction_id, inhibition in drug_effects.items():
            try:
                reaction = self.model.reactions.get_by_id(reaction_id)
                # Assuming inhibition is between 0 (complete inhibition) and 1 (no inhibition)
                new_lower_bound = reaction.lower_bound * inhibition
                new_upper_bound = reaction.upper_bound * inhibition
                reaction.bounds = (new_lower_bound, new_upper_bound)
                logging.info(f"Applied drug effect: {reaction_id} bounds set to {reaction.bounds}")
                print("20Applied drug effect: {reaction_id}")
            except KeyError:
                logging.warning(f"Reaction {reaction_id} not found.")
            except Exception as e:
                logging.error(f"Error applying drug effect to {reaction_id}: {e}")
                raise

        # Apply gene knockouts
        if gene_knockouts:
            for gene_id in gene_knockouts:
                try:
                    gene = self.model.genes.get_by_id(gene_id)
                    cobra.manipulation.delete_model_genes(self.model, [gene])
                    logging.info(f"Knocked out gene: {gene_id}")
                    print("30Knocked out gene: {gene_id}")
                except KeyError:
                    logging.warning(f"Gene {gene_id} not found.")
                except Exception as e:
                    logging.error(f"Error knocking out gene {gene_id}: {e}")
                    raise

        # Apply environmental condition changes
        if condition_changes:
            for exchange_id, flux_value in condition_changes.items():
                try:
                    reaction = self.model.reactions.get_by_id(exchange_id)
                    reaction.lower_bound = flux_value
                    logging.info(f"Environmental condition changed: {exchange_id} lower bound set to {flux_value}")
                    print("40Environmental condition changed: {exchange_id}")
                except KeyError:
                    logging.warning(f"Exchange reaction {exchange_id} not found.")
                except Exception as e:
                    logging.error(f"Error setting environmental condition for {exchange_id}: {e}")
                    raise

        # Perform Flux Variability Analysis (FVA)
        try:
            fva_result = flux_variability_analysis(self.model, fraction_of_optimum=0.9)
            logging.info("Flux Variability Analysis completed.")
            print("50Flux Variability Analysis completed")
        except Exception as e:
            logging.error(f"Error performing Flux Variability Analysis: {e}")
            raise

        # Optimize the model
        try:
            solution = self.model.optimize()
            logging.info(f"Simulated growth rate: {solution.objective_value}")
            print("60Simulated growth rate: {solution.objective_value}")
        except Exception as e:
            logging.error(f"Error optimizing model: {e}")
            raise

        return fva_result, solution



    def apply_perturbations(self, changes: Dict[str, Tuple[float, float]]):
        if self.model is None:
            raise ValueError("No model is loaded.")

        for reaction_id, bounds in changes.items():
            try:
                reaction = self.model.reactions.get_by_id(reaction_id)
                reaction.bounds = bounds
                logging.info(f"Applied perturbation: {reaction_id} new bounds set to {bounds}")
                print("70Applied perturbation: {reaction_id}")
            except KeyError:
                logging.warning(f"Reaction {reaction_id} not found.")
            except Exception as e:
                logging.error(f"Error applying perturbation to {reaction_id}: {e}")
                raise

    def update_metabolite_concentrations(self, solution: Solution, concentrations: dict):

        for reaction in self.model.reactions:
            try:
                flux = solution.fluxes[reaction.id]
                for metabolite, stoichiometry in reaction.metabolites.items():
                    concentrations[metabolite.id] += flux * stoichiometry
                # Logging can be added if needed
            except KeyError:
                logging.warning(f"Reaction {reaction.id} not found in solution fluxes.")
            except Exception as e:
                logging.error(
                    f"Error updating metabolite concentrations for reaction {reaction.id}: {e}"
                )
                raise




    def analyze_hallmarks_of_aging(self, solution: cobra.Solution) -> Dict[str, float]:
        """
        Analyze the hallmarks of aging based on the current metabolic state.
        """
        hallmarks = {
            "genomic_instability": self._analyze_genomic_instability(solution),
            "telomere_attrition": self._analyze_telomere_attrition(solution),
            "epigenetic_alterations": self._analyze_epigenetic_alterations(solution),
            "loss_of_proteostasis": self._analyze_loss_of_proteostasis(solution),
            "deregulated_nutrient_sensing": self._analyze_deregulated_nutrient_sensing(solution),
            "mitochondrial_dysfunction": self._analyze_mitochondrial_dysfunction(solution),
            "cellular_senescence": self._analyze_cellular_senescence(solution),
            "stem_cell_exhaustion": self._analyze_stem_cell_exhaustion(solution),
            "altered_intercellular_communication": self._analyze_altered_intercellular_communication(solution)
        }
        print("80Analyzed hallmarks of aging")
        return hallmarks

    def _analyze_genomic_instability(self, solution: cobra.Solution) -> float:
        # Example: Check flux through DNA repair pathways
        dna_repair_reactions = ["DNA_repair_1", "DNA_repair_2"]  # Replace with actual reaction IDs
        repair_flux = sum(abs(solution.fluxes[rxn]) for rxn in dna_repair_reactions if rxn in solution.fluxes)
        return 1 - min(repair_flux / 10, 1)  # Normalize and invert (higher value = more instability)

    def _analyze_telomere_attrition(self, solution: cobra.Solution) -> float:
        # Example: Check flux through telomerase reaction
        telomerase_reaction = "TELOMERASE"  # Replace with actual reaction ID
        telomerase_flux = abs(solution.fluxes.get(telomerase_reaction, 0))
        return 1 - min(telomerase_flux / 5, 1)  # Normalize and invert

    def _analyze_epigenetic_alterations(self, solution: cobra.Solution) -> float:
        # Example: Check flux through methylation and acetylation reactions
        epigenetic_reactions = ["METHYLATION", "ACETYLATION"]  # Replace with actual reaction IDs
        epigenetic_flux = sum(abs(solution.fluxes[rxn]) for rxn in epigenetic_reactions if rxn in solution.fluxes)
        return 1 - min(epigenetic_flux / 15, 1)  # Normalize and invert

    def _analyze_loss_of_proteostasis(self, solution: cobra.Solution) -> float:
        # Example: Check flux through protein degradation and synthesis pathways
        proteostasis_reactions = ["PROTEIN_SYNTH", "PROTEIN_DEGRAD"]  # Replace with actual reaction IDs
        proteostasis_flux = sum(abs(solution.fluxes[rxn]) for rxn in proteostasis_reactions if rxn in solution.fluxes)
        return 1 - min(proteostasis_flux / 20, 1)  # Normalize and invert

    def _analyze_deregulated_nutrient_sensing(self, solution: cobra.Solution) -> float:
        # Example: Check flux through nutrient uptake reactions
        nutrient_reactions = ["GLUCOSE_UPTAKE", "AMINO_ACID_UPTAKE"]  # Replace with actual reaction IDs
        nutrient_flux = sum(abs(solution.fluxes[rxn]) for rxn in nutrient_reactions if rxn in solution.fluxes)
        return abs(nutrient_flux - 10) / 10  # Deviation from optimal flux

    def _analyze_mitochondrial_dysfunction(self, solution: cobra.Solution) -> float:
        # Example: Check flux through electron transport chain reactions
        etc_reactions = ["NADH_DH", "CYTOCHROME_C_OXID"]  # Replace with actual reaction IDs
        etc_flux = sum(abs(solution.fluxes[rxn]) for rxn in etc_reactions if rxn in solution.fluxes)
        return 1 - min(etc_flux / 30, 1)  # Normalize and invert

    def _analyze_cellular_senescence(self, solution: cobra.Solution) -> float:
        # Example: Check flux through senescence-associated secretory phenotype (SASP) pathways
        sasp_reactions = ["SASP_1", "SASP_2"]  # Replace with actual reaction IDs
        sasp_flux = sum(abs(solution.fluxes[rxn]) for rxn in sasp_reactions if rxn in solution.fluxes)
        return min(sasp_flux / 5, 1)  # Normalize

    def _analyze_stem_cell_exhaustion(self, solution: cobra.Solution) -> float:
        # Example: Check flux through stem cell proliferation pathways
        stem_cell_reactions = ["STEM_CELL_PROLIF", "STEM_CELL_DIFF"]  # Replace with actual reaction IDs
        stem_cell_flux = sum(abs(solution.fluxes[rxn]) for rxn in stem_cell_reactions if rxn in solution.fluxes)
        return 1 - min(stem_cell_flux / 10, 1)  # Normalize and invert

    def _analyze_altered_intercellular_communication(self, solution: cobra.Solution) -> float:
        # Example: Check flux through signaling pathways
        signaling_reactions = ["CYTOKINE_PROD", "GROWTH_FACTOR_SIGNAL"]  # Replace with actual reaction IDs
        signaling_flux = sum(abs(solution.fluxes[rxn]) for rxn in signaling_reactions if rxn in solution.fluxes)
        return abs(signaling_flux - 15) / 15  # Deviation from optimal flux

    def visualize_hallmarks_of_aging(self, hallmarks: Dict[str, float]):
        """
        Visualize the hallmarks of aging using a radar chart.
        """
        labels = list(hallmarks.keys())
        values = list(hallmarks.values())

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # Repeat the first value to close the polygon
        angles = np.concatenate((angles, [angles[0]]))  # Repeat the first angle to close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
        ax.set_title("Hallmarks of Aging")
        ax.set_ylim(0, 1)
        plt.show()
        print("90Visualized hallmarks of aging")

    def simulate_aging(self, time_points: List[float], perturbations: Optional[List[Dict]] = None) -> Dict[float, Dict]:
        """
        Simulate aging over time and analyze hallmarks at each time point.
        """
        aging_results = {}
        for t in time_points:
            if perturbations:
                self.apply_perturbations(perturbations[t])
            
            solution = self.model.optimize()
            hallmarks = self.analyze_hallmarks_of_aging(solution)
            
            aging_results[t] = {
                "growth_rate": solution.objective_value,
                "hallmarks": hallmarks
            }
            print("95Simulated aging at time point {t}")
        
        return aging_results





    def analyze_pathways(self, solution: Solution, threshold: float = 0.1) -> Dict[str, bool]:

        pathway_activities = {}
        for group in self.model.groups:
            try:
                pathway_flux = sum(
                    abs(solution.fluxes.get(r.id, 0.0)) for r in group.members
                )
                pathway_activities[group.id] = pathway_flux > threshold
                logging.info(
                    f"Pathway {group.id} active: {pathway_activities[group.id]}"
                )
            except Exception as e:
                logging.error(f"Error analyzing pathway {group.id}: {e}")
                raise
        print("100Analyzed pathway activities")
        return pathway_activities

    def simulate_time_series(
        self,
        initial_conditions: Dict[str, Tuple[float, float]],
        perturbations: List[Dict],
        time_points: List[float],
    ) -> Dict[float, Dict]:
        if self.model is None:
            raise ValueError("No model is loaded.")

        results = {}
        metabolite_concentrations = {m.id: 0.0 for m in self.model.metabolites}
        logging.info("Initial metabolite concentrations set.")
        print("10Initial metabolite concentrations set")

        # Reset model to initial conditions
        self.reset_model()
        logging.info("Model reset to initial conditions.")
        print("20Model reset to initial conditions")

        # Apply initial conditions
        for reaction_id, bounds in initial_conditions.items():
            try:
                reaction = self.model.reactions.get_by_id(reaction_id)
                reaction.bounds = bounds
                logging.info(f"Set initial condition for {reaction_id}: {bounds}")
            except KeyError:
                logging.warning(f"Reaction {reaction_id} not found.")
            except Exception as e:
                logging.error(f"Error setting initial condition for {reaction_id}: {e}")
                raise
        print("30Applied initial conditions")

        perturbation_dict = {p["time"]: p["changes"] for p in perturbations}

        for t in time_points:
            logging.info(f"Simulating time point: {t}")
            print("40 + int(50 * (t / time_points[-1]))Simulating time point: {t}")

            # Apply perturbations if any at this time point
            if t in perturbation_dict:
                self.apply_perturbations(perturbation_dict[t])

            # Optimize the model
            try:
                solution = self.model.optimize()
                logging.info(
                    f"Optimized model at time {t}, growth rate: {solution.objective_value}"
                )
            except Exception as e:
                logging.error(f"Error optimizing model at time {t}: {e}")
                raise

            # Update metabolite concentrations
            self.update_metabolite_concentrations(solution, metabolite_concentrations)
            logging.info(f"Updated metabolite concentrations at time {t}")

            # Analyze pathways
            pathway_activities = self.analyze_pathways(solution)
            logging.info(f"Analyzed pathway activities at time {t}")

            # Store results
            results[t] = {
                "growth_rate": solution.objective_value,
                "fluxes": solution.fluxes.to_dict(),
                "metabolite_concentrations": metabolite_concentrations.copy(),
                "pathway_activities": pathway_activities,
            }

        print("100Time series simulation completed")
        return results

    def sensitivity_analysis(
        self, parameter_ranges: Dict[str, np.ndarray], objective: str = "objective_value"
    ) -> Dict[str, List[float]]:

        if self.model is None:
            raise ValueError("No model is loaded.")

        sensitivities = {}
        for parameter, range_values in parameter_ranges.items():
            parameter_sensitivities = []
            for value in range_values:
                with self.model:
                    try:
                        # Assuming parameter is a reaction bound
                        reaction = self.model.reactions.get_by_id(parameter)
                        reaction.bounds = (value, value)  # Fix the reaction flux to the value
                        solution = self.model.optimize()
                        parameter_sensitivities.append(getattr(solution, objective))
                        logging.info(
                            f"Parameter {parameter} set to {value}, result: {getattr(solution, objective)}"
                        )
                    except Exception as e:
                        logging.error(
                            f"Error during sensitivity analysis for {parameter} with value {value}: {e}"
                        )
                        parameter_sensitivities.append(None)
            sensitivities[parameter] = parameter_sensitivities
            print("100Completed sensitivity analysis for {parameter}")
        return sensitivities




    def train_ml_model(self, simulation_results: List[Dict]) -> RandomForestRegressor:
        X = []  # Features (e.g., perturbations, conditions)
        y = []  # Target (e.g., growth rate)

        for result in simulation_results:
            X.append(result["conditions"])
            y.append(result["growth_rate"])

        try:
            X = pd.DataFrame(X)
            y = pd.Series(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            logging.info("Machine learning model trained successfully.")
            print("Machine learning model trained successfully")

            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model Mean Squared Error: {mse}")
            logging.info(f"Model R² Score: {r2}")
            print(f"Model evaluation - MSE: {mse}, R²: {r2}")

            # Save the trained model
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)
            print("Trained model saved to file")

        except Exception as e:
            logging.error(f"Error training machine learning model: {e}")
            raise

        return model

    def visualize_fluxes(self, solution: cobra.Solution):
        fluxes = solution.fluxes
        flux_data = pd.DataFrame({"Reaction": fluxes.index, "Flux": fluxes.values})

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            flux_data.set_index("Reaction").T, cmap="viridis", cbar=True, annot=True
        )
        plt.title("Flux Distribution Heatmap")
        plt.xlabel("Reactions")
        plt.ylabel("Flux Value")
        plt.show()
        print("Visualized flux distribution")

    def visualize_growth_rate(self, time_series_results: Dict[float, Dict]):
        times = sorted(time_series_results.keys())
        growth_rates = [time_series_results[t]["growth_rate"] for t in times]

        plt.figure(figsize=(10, 5))
        plt.plot(times, growth_rates, marker="o", linestyle="-")
        plt.title("Growth Rate Over Time")
        plt.xlabel("Time")
        plt.ylabel("Growth Rate")
        plt.grid(True)
        plt.show()
        print("Visualized growth rate over time")

    def plot_sensitivity_analysis(self, sensitivity_results: Dict[str, List[float]]):
        plt.figure(figsize=(10, 6))
        for parameter, sensitivities in sensitivity_results.items():
            plt.plot(sensitivities, label=parameter)
        plt.title("Sensitivity Analysis")
        plt.xlabel("Parameter Value Index")
        plt.ylabel("Objective (e.g., Growth Rate)")
        plt.legend()
        plt.grid(True)
        plt.show()
        print("Plotted sensitivity analysis results")

    def emit_progress(self, phase: str, progress: int, message: str):
        """Emit progress updates."""
        print(f"Progress: {phase} - {progress}% - {message}")


# Example usage
if __name__ == "__main__":
    simulator = DigitalTwinSimulator(r"C:\Users\wes\AutoProtGenerationSystem\Phase_3\aging_model.json")

    # Example: Simulate inhibition of multiple reactions and environmental changes
    drug_effects = {
        "RXN_ID_1": 0.5,  # 50% inhibition
        "RXN_ID_2": 0.0,  # Complete inhibition
    }
    gene_knockouts = ["gene_1", "gene_2"]  # Example gene knockouts
    condition_changes = {
        "EX_glc__D_e": -10.0,
        "EX_o2_e": -20.0,
    }  # Changes in glucose and oxygen uptake

    fva_result, growth_rate = simulator.simulate_cellular_environment(
        drug_effects, gene_knockouts, condition_changes
    )

    # Visualize flux distribution
    solution = simulator.model.optimize()
    simulator.visualize_fluxes(solution)

    # Example: Time-series simulation
    initial_conditions = {r.id: r.bounds for r in simulator.model.reactions}
    perturbations = [
        {"time": 0, "changes": {"RXN_ID_1": (0, 10)}},
        {"time": 5, "changes": {"RXN_ID_2": (0, 5)}},
        {"time": 10, "changes": {"EX_glc__D_e": (-5.0, -5.0)}},
    ]
    time_points = list(range(0, 15))

    time_series_results = simulator.simulate_time_series(
        initial_conditions, perturbations, time_points
    )

    # Visualize growth rate over time
    simulator.visualize_growth_rate(time_series_results)

    # Example: Sensitivity analysis
    parameter_ranges = {
        "RXN_ID_1": np.linspace(0, 10, 10),
        "RXN_ID_2": np.linspace(0, 5, 10),
    }
    sensitivity_results = simulator.sensitivity_analysis(parameter_ranges)

    # Plot sensitivity analysis
    simulator.plot_sensitivity_analysis(sensitivity_results)

    # Example: Collect simulation results for ML model training
    simulation_data = [
        {"conditions": {"RXN_ID_1": 0.5, "RXN_ID_2": 0.0}, "growth_rate": 0.7},
        {"conditions": {"RXN_ID_1": 0.8, "RXN_ID_2": 0.2}, "growth_rate": 0.9},
        # Add more simulated data...
    ]

    # Analyze hallmarks of aging
    solution = simulator.model.optimize()
    hallmarks = simulator.analyze_hallmarks_of_aging(solution)
    simulator.visualize_hallmarks_of_aging(hallmarks)

    # Simulate aging over time
    time_points = list(range(0, 100, 10))  # 0 to 90, step 10
    aging_results = simulator.simulate_aging(time_points)

    # Visualize aging progression
    for t, result in aging_results.items():
        print(f"Time: {t}")
        print(f"Growth Rate: {result['growth_rate']}")
        simulator.visualize_hallmarks_of_aging(result['hallmarks'])

    print("Aging simulation completed successfully!")



    # Train the ML model
    trained_model = simulator.train_ml_model(simulation_data)

    print("Simulation completed successfully!")
