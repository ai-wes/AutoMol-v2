import cobra
from cobra import Model, Reaction, Metabolite
from cobra.io import save_json_model
from Phase_3.d_twin import DigitalTwinSimulator
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

def set_objective_and_optimize(model):
    """
    Sets the objective function and optimizes the model.

    Args:
        model (cobra.Model): The metabolic model.
    """
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
    """
    Creates a metabolic model and saves it to a JSON file.

    Args:
        json_path (str): Path to save the JSON model.
    """
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

def main():
    """
    Main function to create and save the metabolic model.
    """
    setup_logging()
    logging.info("Starting the metabolic model creation process.")
    json_path = 'aging_model.json'
    create_and_save_metabolic_model(json_path)
    logging.info("Metabolic model creation process completed.")

if __name__ == '__main__':
    main()