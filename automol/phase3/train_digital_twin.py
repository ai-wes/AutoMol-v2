import json
import matplotlib.pyplot as plt
from typing import Dict, List
import cobra
import os
import sys

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from cobra import Model, Reaction, Metabolite
from cobra.io import save_json_model
from phase3.digital_twin import DigitalTwinSimulator
import logging
import sys

def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('train_digital_twin.log')
        ]
    )

def create_metabolites(model):
    """
    Creates and adds metabolites to the model.

    Args:
        model (cobra.Model): The metabolic model.
    
    Returns:
        dict: A dictionary of created metabolites.
        """
    metabolites = {}
    metabolites['atg_proteins_c'] = Metabolite('atg_proteins_c', name='ATG Proteins', compartment='c')
    metabolites['damaged_proteins_c'] = Metabolite('damaged_proteins_c', name='Damaged Proteins', compartment='c')
    metabolites['amino_acids_c'] = Metabolite('amino_acids_c', name='Amino Acids', compartment='c')
    metabolites['telomerase_rna_c'] = Metabolite('telomerase_rna_c', name='Telomerase RNA Component', compartment='c')
    metabolites['telomerase_protein_c'] = Metabolite('telomerase_protein_c', name='Telomerase Protein Component', compartment='c')
    metabolites['telomerase_complex_c'] = Metabolite('telomerase_complex_c', name='Telomerase Complex', compartment='c')
    metabolites['telomere_units_c'] = Metabolite('telomere_units_c', name='Telomere Units', compartment='c')
    metabolites['nucleotides_c'] = Metabolite('nucleotides_c', name='Nucleotides', compartment='c')
    metabolites['energy_c'] = Metabolite('energy_c', name='Energy (ATP)', compartment='c')
    metabolites['substrate_c'] = Metabolite('substrate_c', name='Energy Substrate', compartment='c')
    metabolites['biomass_c'] = Metabolite('biomass_c', name='Biomass', compartment='c')
    
    for metabolite in metabolites.values():
        model.add_metabolites([metabolite])
        logging.debug(f"Added metabolite: {metabolite.id}")
    
    return metabolites

def create_reactions(model, metabolites):
    """
    Creates and adds reactions to the model.

    Args:
        model (cobra.Model): The metabolic model.
        metabolites (dict): Dictionary of metabolites.
    """
    # Autophagy reactions
    rxn_autophagosome_formation = Reaction('AUTOPHAGY_FORMATION')
    rxn_autophagosome_formation.name = 'Autophagosome Formation'
    rxn_autophagosome_formation.lower_bound = 0.0
    rxn_autophagosome_formation.upper_bound = 1000.0
    rxn_autophagosome_formation.add_metabolites({
        metabolites['atg_proteins_c']: -1.0,
        metabolites['energy_c']: -1.0,
        metabolites['damaged_proteins_c']: -1.0,
    })
    model.add_reactions([rxn_autophagosome_formation])
    logging.debug(f"Added reaction: {rxn_autophagosome_formation.id}")

    rxn_autophagic_degradation = Reaction('AUTOPHAGIC_DEGRADATION')
    rxn_autophagic_degradation.name = 'Autophagic Degradation'
    rxn_autophagic_degradation.lower_bound = 0.0
    rxn_autophagic_degradation.upper_bound = 1000.0
    rxn_autophagic_degradation.add_metabolites({
        metabolites['damaged_proteins_c']: -1.0,
        metabolites['energy_c']: -1.0,
        metabolites['amino_acids_c']: 1.0,
        metabolites['atg_proteins_c']: 1.0,
    })
    model.add_reactions([rxn_autophagic_degradation])
    logging.debug(f"Added reaction: {rxn_autophagic_degradation.id}")

    # Telomerase activity reactions
    rxn_telomerase_assembly = Reaction('TELOMERASE_ASSEMBLY')
    rxn_telomerase_assembly.name = 'Telomerase Assembly'
    rxn_telomerase_assembly.lower_bound = 0.0
    rxn_telomerase_assembly.upper_bound = 1000.0
    rxn_telomerase_assembly.add_metabolites({
        metabolites['telomerase_rna_c']: -1.0,
        metabolites['telomerase_protein_c']: -1.0,
        metabolites['energy_c']: -1.0,
        metabolites['telomerase_complex_c']: 1.0,
    })
    model.add_reactions([rxn_telomerase_assembly])
    logging.debug(f"Added reaction: {rxn_telomerase_assembly.id}")

    rxn_telomere_elongation = Reaction('TELOMERE_ELONGATION')
    rxn_telomere_elongation.name = 'Telomere Elongation'
    rxn_telomere_elongation.lower_bound = 0.0
    rxn_telomere_elongation.upper_bound = 1000.0
    rxn_telomere_elongation.add_metabolites({
        metabolites['telomerase_complex_c']: -1.0,
        metabolites['nucleotides_c']: -1.0,
        metabolites['telomere_units_c']: 1.0,
        metabolites['energy_c']: -1.0,
    })
    model.add_reactions([rxn_telomere_elongation])
    logging.debug(f"Added reaction: {rxn_telomere_elongation.id}")

    # Energy production reaction
    rxn_energy_production = Reaction('ENERGY_PRODUCTION')
    rxn_energy_production.name = 'Energy Production'
    rxn_energy_production.lower_bound = 0.0
    rxn_energy_production.upper_bound = 1000.0
    rxn_energy_production.add_metabolites({
        metabolites['substrate_c']: -1.0,
        metabolites['energy_c']: 1.0,
    })
    model.add_reactions([rxn_energy_production])
    logging.debug(f"Added reaction: {rxn_energy_production.id}")

    # Biomass reaction (objective)
    biomass = Reaction('BIOMASS')
    biomass.name = 'Biomass Production'
    biomass.lower_bound = 0.0
    biomass.upper_bound = 1000.0
    biomass.add_metabolites({
        metabolites['amino_acids_c']: -1.0,
        metabolites['nucleotides_c']: -1.0,
        metabolites['energy_c']: -30.0,
        metabolites['biomass_c']: 1.0,
    })
    model.add_reactions([biomass])
    logging.debug(f"Added reaction: {biomass.id}")

    # Exchange reactions with increased uptake rates
    exchange_reactions = {
        'EX_substrate': ('Exchange Substrate', metabolites['substrate_c']),
        'EX_amino_acids': ('Exchange Amino Acids', metabolites['amino_acids_c']),
        'EX_nucleotides': ('Exchange Nucleotides', metabolites['nucleotides_c']),
    }

    for rxn_id, (name, metabolite) in exchange_reactions.items():
        rxn = Reaction(rxn_id)
        rxn.name = name
        rxn.lower_bound = -1000.0  # Increased uptake
        rxn.upper_bound = 1000.0
        rxn.add_metabolites({
            metabolite: -1.0,
        })
        model.add_reactions([rxn])
        logging.debug(f"Added exchange reaction: {rxn.id}")

    # Demand reaction for biomass_c
    demand_biomass = Reaction('DM_biomass_c')
    demand_biomass.name = 'Demand Biomass'
    demand_biomass.lower_bound = 0.0
    demand_biomass.upper_bound = 1000.0
    demand_biomass.add_metabolites({
        metabolites['biomass_c']: -1.0,
    })
    model.add_reactions([demand_biomass])
    logging.debug(f"Added demand reaction: {demand_biomass.id}")


    # Add new reactions for protein synthesis and degradation
    rxn_protein_synthesis = Reaction('PROTEIN_SYNTHESIS')
    rxn_protein_synthesis.name = 'Protein Synthesis'
    rxn_protein_synthesis.lower_bound = 0.0
    rxn_protein_synthesis.upper_bound = 1000.0
    rxn_protein_synthesis.add_metabolites({
        metabolites['amino_acids_c']: -4.0,
        metabolites['energy_c']: -4.0,
        metabolites['atg_proteins_c']: 1.0,
    })
    model.add_reactions([rxn_protein_synthesis])
    logging.debug(f"Added reaction: {rxn_protein_synthesis.id}")

    rxn_protein_degradation = Reaction('PROTEIN_DEGRADATION')
    rxn_protein_degradation.name = 'Protein Degradation'
    rxn_protein_degradation.lower_bound = 0.0
    rxn_protein_degradation.upper_bound = 1000.0
    rxn_protein_degradation.add_metabolites({
        metabolites['atg_proteins_c']: -1.0,
        metabolites['energy_c']: -1.0,
        metabolites['amino_acids_c']: 3.5,
    })
    model.add_reactions([rxn_protein_degradation])
    logging.debug(f"Added reaction: {rxn_protein_degradation.id}")






def plot_aging_metrics(metrics: Dict[str, List[float]], output_path: str):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Aging Metrics Over Time')

    for ax, (key, values) in zip(axs.ravel(), metrics.items()):
        ax.plot(values)
        ax.set_title(key.replace('_', ' ').title())
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Aging metrics plot saved to {output_path}")



def set_objective_and_optimize(model):

    model.objective = 'BIOMASS'
    logging.info("Objective function set to 'BIOMASS'.")

    try:
        solution = model.optimize()
        if solution.status != 'optimal':
            logging.warning("The model optimization did not find an optimal solution.")
        else:
            logging.info(f"Objective value: {solution.objective_value}")
            logging.info(f"Fluxes:\n{solution.fluxes}")
    except Exception as e:
        logging.error(f"Optimization failed: {e}")



def create_and_save_metabolic_model(json_path: str):

    logging.info("Initializing a new metabolic model.")
    model = Model('aging_metabolic_model')

    logging.info("Creating metabolites.")
    metabolites = create_metabolites(model)

    logging.info("Creating reactions.")
    create_reactions(model, metabolites)

    logging.info("Setting objective and optimizing the model.")
    set_objective_and_optimize(model)

    logging.info(f"Saving the model to '{json_path}'.")
    try:
        save_json_model(model, json_path)
        logging.info(f"Model successfully saved to '{json_path}'.")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")


def load_model(json_path: str) -> Model:
    try:
        model = cobra.io.load_json_model(json_path)
        logging.info(f"Model successfully loaded from '{json_path}'.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from '{json_path}': {e}")
        raise

def simulate_aging(model: Model, num_iterations: int = 100) -> Dict[str, List[float]]:
    metrics = {
        'biomass': [],
        'telomere_length': [],
        'autophagy_rate': [],
        'energy_production': []
    }

    for i in range(num_iterations):
        # Simulate telomere shortening
        telomere_reaction = model.reactions.get_by_id('TELOMERE_ELONGATION')
        telomere_reaction.upper_bound *= 0.99  # Decrease telomere elongation capacity

        # Simulate increased protein damage
        autophagy_reaction = model.reactions.get_by_id('AUTOPHAGY_FORMATION')
        autophagy_reaction.lower_bound *= 1.01  # Increase minimum required autophagy

        # Optimize the model
        solution = model.optimize()

        # Track metrics
        metrics['biomass'].append(solution.objective_value)
        metrics['telomere_length'].append(telomere_reaction.upper_bound)
        metrics['autophagy_rate'].append(solution.fluxes['AUTOPHAGY_FORMATION'])
        metrics['energy_production'].append(solution.fluxes['ENERGY_PRODUCTION'])

        logging.info(f"Iteration {i+1}/{num_iterations}: Biomass = {solution.objective_value:.4f}")

    return metrics

def main():
    setup_logging()
    logging.info("Starting the metabolic model creation and aging simulation process.")

    json_path = 'new_aging_model.json'
    create_and_save_metabolic_model(json_path)

    # Load the saved model
    model = load_model(json_path)

    # Simulate aging process
    logging.info("Starting aging simulation.")
    aging_metrics = simulate_aging(model, num_iterations=100)

    # Plot and save aging metrics
    plot_aging_metrics(aging_metrics, 'aging_metrics.png')

    # Save the final aged model
    aged_model_path = 'new_aging_model.json'
    save_json_model(model, aged_model_path)
    logging.info(f"Aged model saved to {aged_model_path}")

    logging.info("Metabolic model creation and aging simulation process completed.")

if __name__ == '__main__':
    main()