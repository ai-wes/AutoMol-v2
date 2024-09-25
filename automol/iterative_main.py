def main(iterations: int = 3, top_n: int = 5, score_threshold: float = 0.8, improvement_threshold: float = 0.05):
    """
    Main function to run the molecular design pipeline iteratively.

    Args:
        iterations (int): Number of iterative runs.
        top_n (int): Number of top molecules to select each iteration.
        score_threshold (float): Minimum score required to be considered promising.
        improvement_threshold (float): Minimum relative improvement required to continue iterations.
    """
    # Load configuration
    args = parse_arguments()
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logger.error("config.json file not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config.json.")
        sys.exit(1)

    # Merge config with arguments
    config = merge_config_with_args(config, args)
    logger.info("Configuration merged with command-line arguments.")

    config_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(config_overrides)
    logger.info(f"Configuration overridden with command-line arguments: {config_overrides}")

    # Initialize variables to hold all results across iterations
    cumulative_results = {
        'phase1': [],
        'phase2a': [],
        'phase2b': [],
        'phase3': [],
        'phase4': []
    }

    previous_best_score = 0  # Initialize previous best score

    # Iterate the pipeline
    for iteration in range(1, iterations + 1):
        logger.info(f"=== Starting Iteration {iteration} ===")
        print(f"\n=== Starting Iteration {iteration} ===\n")

        # Create organized directory structure for this iteration
        base_output_dir = config.get('base_output_dir', 'results')
        iteration_output_dir = os.path.join(base_output_dir, f"iteration_{iteration}")
        try:
            run_dir, phase_dirs = create_organized_directory_structure(iteration_output_dir)
            if not run_dir or not phase_dirs:
                raise ValueError("Failed to create directory structure")
            logger.info(f"Organized directory structure created at {run_dir}.")
        except Exception as e:
            logger.error(f"Failed to create directory structure: {str(e)}")
            sys.exit(1)

        # Update config with run_dir
        config['run_dir'] = run_dir

        # Phase 1: Generate Hypothesis
        if not config.get('skip_description_gen', False):
            logger.info("Starting Phase 1: Generate Hypothesis")
            phase1_results = run_Phase_1(config)
            save_json(phase1_results, Path(run_dir) / "phase1_results.json")
            logger.info("Phase 1 results saved successfully.")
        else:
            logger.info("Skipping Phase 1: Generate Hypothesis as per command-line flag.")
            # Optionally, set default values or handle accordingly
            phase1_results = {
                'technical_description': config.get('input_text', ''),
                'initial_hypothesis': config.get('input_text', '')
            }
        print("phase1_results", phase1_results)

        # Phase 2a: Generate and Optimize Proteins
        logger.info("Starting Phase 2a: Generate and Optimize Proteins")
        phase2a_config = config['phase2a']
        phase2a_config.update({
            'technical_descriptions': [phase1_results['technical_description']],  # Ensure it's a list
            'predicted_structures_dir': os.path.join(run_dir, "phase2a", "generated_sequences"),
            'results_dir': os.path.join(run_dir, "phase2a", "results"),
            'num_sequences': config.get('num_sequences', 2),
            'optimization_steps': config.get('optimization_steps', 15),
            'score_threshold': config.get('score_threshold', 0.55)
        })

        # Filter out unexpected keys
        expected_keys = ['technical_descriptions', 'predicted_structures_dir', 'results_dir', 'num_sequences', 'optimization_steps', 'score_threshold']
        filtered_phase2a_config = {k: v for k, v in phase2a_config.items() if k in expected_keys}

        # Unpack the filtered configuration dictionary when calling run_Phase_2a
        phase2a_results, all_generated_sequences = run_Phase_2a(**filtered_phase2a_config)
        save_json(phase2a_results, Path(run_dir) / "phase2a_results.json")
        logger.info("Phase 2a results saved successfully.")

        # Extract protein sequences from phase2a_results
        protein_sequences = [result['sequence'] for result in phase2a_results]

        # Phase 2b: Generate and Optimize Ligands
        logger.info("Starting Phase 2b: Generate and Optimize Ligands")
        phase2b_config = config['phase2b']
        phase2b_config.update({
            'predicted_structures_dir': os.path.join(run_dir, "phase2b", "ligands"),  # Corrected path
            'results_dir': os.path.join(run_dir, "phase2b", "results"),
            'num_sequences': config.get('num_sequences', 2),
            'optimization_steps': config.get('optimization_steps', 15),
            'score_threshold': config.get('score_threshold', 0.55),
            'protein_sequences': protein_sequences
        })

        # Filter out unexpected keys
        expected_keys_phase2b = ['predicted_structures_dir', 'results_dir', 'num_sequences', 'optimization_steps', 'score_threshold', 'protein_sequences']
        filtered_phase2b_config = {k: v for k, v in phase2b_config.items() if k in expected_keys_phase2b}

        phase2b_results = run_Phase_2b(**filtered_phase2b_config)
        save_json(phase2b_results, Path(run_dir) / "phase2b_results.json")
        logger.info("Phase 2b results saved successfully.")

        # Phase 3: Simulation
        logger.info("Starting Phase 3: Simulation")
        phase3_config = config['phase3']
        phase3_config.update({
            'protein_results': phase2a_results,
            'output_dir': os.path.join(run_dir, "phase3"),
            'device': config.get('device', 'cpu')
        })

        # Optionally, filter out unexpected keys for Phase 3
        expected_keys_phase3 = ['protein_results', 'output_dir', 'device']
        filtered_phase3_config = {k: v for k, v in phase3_config.items() if k in expected_keys_phase3}

        # Unpack the filtered configuration dictionary when calling run_Phase_3
        simulation_results = run_Phase_3(**filtered_phase3_config)
        save_json(simulation_results, Path(run_dir) / "phase3" / "phase3_results.json")
        logger.info("Phase 3 results saved successfully.")

        # Phase 4: Final Analysis and Reporting
        logger.info("Starting Phase 4: Final Analysis and Reporting")
        phase4_config = config['phase4']
        phase4_config.update({
            'simulation_results': simulation_results,
            'output_dir': os.path.join(run_dir, "phase4")
        })

        phase4_results = run_Phase_4(simulation_results, phase4_config)  # Pass simulation_results instead of phase3_results
        save_json(phase4_results, Path(run_dir) / "phase4_results.json")
        logger.info("Phase 4 results saved successfully.")

        # Save All Results Consolidated
        all_results = {
            'phase1': phase1_results,
            'phase2a': phase2a_results,
            'phase2b': phase2b_results,
            'phase3': simulation_results,
            'phase4': phase4_results
        }
        save_json(all_results, Path(run_dir) / "final_results.json")
        logger.info("All phase results saved successfully.")

        # Append to cumulative results
        for phase in cumulative_results.keys():
            cumulative_results[phase].append(all_results.get(phase, {}))

        # Selection of promising molecules for next iteration
        promising_molecules = select_promising_molecules(all_results, top_n=top_n, score_threshold=score_threshold)
        logger.info(f"Selected {len(promising_molecules)} promising molecules for next iteration.")
        print(f"Selected {len(promising_molecules)} promising molecules for next iteration.")

        if not promising_molecules:
            logger.info("No promising molecules found. Ending iterations.")
            print("No promising molecules found. Ending iterations.")
            break  # Exit the loop if no promising molecules are found

        # Update configuration for the next iteration
        # For example, you might want to adjust parameters based on the findings
        # Here, we'll assume the same configuration is used
        # If you need to modify the config, do it here

        # Check for significant improvement
        current_best_score = max([mol.get('score', 0) for mol in promising_molecules], default=0)
        if has_significant_improvement(current_best_score, previous_best_score, improvement_threshold):
            logger.info(f"Significant improvement detected: {current_best_score:.2f} > {previous_best_score:.2f}")
            print(f"Significant improvement detected: {current_best_score:.2f} > {previous_best_score:.2f}")
            previous_best_score = current_best_score
        else:
            logger.info("No significant improvement detected. Ending iterations.")
            print("No significant improvement detected. Ending iterations.")
            break

    # After all iterations, optionally generate a final consolidated report
    generate_final_report(cumulative_results, base_output_dir)
    logger.info("Iterative pipeline completed.")
    print("Iterative pipeline completed.")
