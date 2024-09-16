from Phase_3.d_twin import DigitalTwinSimulator
from cobra.core import Metabolite, Reaction
import numpy as np
import pandas as pd

def main():
    # Initialize the simulator with the path to your new JSON model
    simulator = DigitalTwinSimulator('aging_model.json')
    model = simulator.model


    # Load the CSV data
    file_path = 'your_csv_file.csv'  # Replace with the actual path to your CSV file
    data = pd.read_csv(file_path)

    # Display the first few rows to inspect the data structure
    print(data.head())

    # Extract relevant sample details from the 'Samples' column
    samples = data['Samples'].iloc[0]  # Adjust the index if necessary to target the correct row
    sample_details = eval(samples)  # Convert string representation of list to actual list

    # Print out sample details for verification
    for sample in sample_details:
        print(f"Accession: {sample['Accession']}, Title: {sample['Title']}")

    # Define gene knockdowns based on sample titles
    gene_knockdowns = []
    for sample in sample_details:
        if 'shTp53' in sample['Title']:
            gene_knockdowns.append('Tp53')
        if 'shNF1' in sample['Title']:
            gene_knockdowns.append('NF1')
        if 'shCBS' in sample['Title']:
            gene_knockdowns.append('CBS')

    # Remove duplicates from the gene knockdown list
    gene_knockdowns = list(set(gene_knockdowns))

    # Define drug effects based on your research focus
    drug_effects = {
        'TERT_activation': 1.0  # Maintain TERT activation as per the experimental setup
    }

    # Define environmental condition changes
    condition_changes = {
        'EX_glucose': -10.0  # Example condition change; adjust as per your needs
    }

    # Initialize the Digital Twin Simulator with your JSON model
    simulator = DigitalTwinSimulator('aging_model.json')

    # Run the simulation with extracted knockdown conditions
    fva_result, solution = simulator.simulate_cellular_environment(
        drug_effects=drug_effects,
        gene_knockouts=gene_knockdowns,
        condition_changes=condition_changes
    )

    # Output simulation results
    print(f"Growth rate after simulation: {solution.objective_value}")
    print(f"FVA Results:\n{fva_result}")

    # Define RNA-seq results for validation
    rna_seq_results = {
        'Tp53': {'effect': 'down', 'impact': 'reduced senescence'},
        'NF1': {'effect': 'down', 'impact': 'altered AIS maintenance'},
        'CBS': {'effect': 'down', 'impact': 'unknown'}
    }

    # Validate and compare simulation results with RNA-seq findings
    for gene, results in rna_seq_results.items():
        simulated_flux = fva_result.loc[gene, 'minimum'] if gene in fva_result.index else 'N/A'
        print(f"Gene: {gene}, RNA-seq Effect: {results['effect']}, Simulated Flux: {simulated_flux}")

    # Define and run time-series simulation based on initial conditions
    initial_conditions = {r.id: r.bounds for r in simulator.model.reactions}
    perturbations = [
        {'time': 5, 'changes': {'AUTOPHAGY_FORMATION': (0.0, 0.0)}},  # Example perturbation
        {'time': 10, 'changes': {'TELOMERASE_ASSEMBLY': (0.0, 2000.0)}},  # Activate telomerase at time 10
    ]

    time_points = list(range(0, 15))

    # Run time-series simulation
    time_series_results = simulator.simulate_time_series(
        initial_conditions=initial_conditions,
        perturbations=perturbations,
        time_points=time_points
    )

    # Visualize growth rate changes over time
    simulator.visualize_growth_rate(time_series_results)

    # Sensitivity analysis on parameters
    parameter_ranges = {
        'ENERGY_PRODUCTION': np.linspace(500, 1500, 5),
        'AUTOPHAGY_FORMATION': np.linspace(0, 1000, 5),
    }

    # Perform sensitivity analysis
    sensitivity_results = simulator.sensitivity_analysis(parameter_ranges)

    # Plot sensitivity analysis results
    simulator.plot_sensitivity_analysis(sensitivity_results)

    # Example of training a machine learning model with simulated data
    simulation_data = [
        {'conditions': {'AUTOPHAGY_FORMATION': 0.0, 'TELOMERASE_ASSEMBLY': 1000.0}, 'growth_rate': 35.0},
        {'conditions': {'AUTOPHAGY_FORMATION': 500.0, 'TELOMERASE_ASSEMBLY': 500.0}, 'growth_rate': 40.0},
        # Add additional simulated data points as needed
    ]

    # Train the ML model with simulation data
    trained_model = simulator.train_ml_model(simulation_data)

    # Adding oxidative stress reaction to the model
    ros_c = Metabolite('ros_c', name='Reactive Oxygen Species', compartment='c')

    # Ensure 'energy_c' exists in the model
    try:
        energy_c = simulator.model.metabolites.get_by_id('energy_c')
    except KeyError:
        raise KeyError("Metabolite 'energy_c' not found in the model.")

    rxn_ros_production = Reaction('ROS_PRODUCTION')
    rxn_ros_production.name = 'ROS Production'
    rxn_ros_production.lower_bound = 0.0
    rxn_ros_production.upper_bound = 1000.0
    rxn_ros_production.add_metabolites({
        energy_c: -1.0,
        ros_c: 1.0,
    })

    # Add ROS production reaction to the model
    simulator.model.add_reactions([rxn_ros_production])

    # Update original bounds after adding the ROS production reaction
    simulator.original_bounds = {r.id: r.bounds for r in simulator.model.reactions}

    # Re-optimize the model after adding the ROS production reaction
    solution = simulator.model.optimize()
    print(f"New growth rate after adding ROS_PRODUCTION: {solution.objective_value}")

    # Validate new simulation results
    fva_result, solution = simulator.simulate_cellular_environment(
        drug_effects=drug_effects,
        gene_knockouts=gene_knockdowns,
        condition_changes=condition_changes
    )

    # Final output of simulation results
    print(f"Final growth rate: {solution.objective_value}")
    print(f"Final FVA Results:\n{fva_result}")

if __name__ == '__main__':
    main()
