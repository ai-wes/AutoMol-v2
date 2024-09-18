# AutoMol-v2/automol/phase3/run_Phase_3.py

import random
import os
import sys
import logging
from pymol import cmd
from phase3.digital_twin import DigitalTwinSimulator
import multiprocessing
from subprocess import run
from phase3.analyze import run_analysis_pipeline
from phase3.simulate import  run_simulation_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyMOL Analysis Function
def run_pymol_analysis(structure_file, trajectory_file, output_dir):
    try:
        logging.info("Initializing PyMOL session")
        cmd.reinitialize()
        
        logging.info(f"Loading structure file: {structure_file}")
        cmd.load(structure_file, 'structure')
        
        logging.info(f"Loading trajectory file: {trajectory_file}")
        cmd.load_traj(trajectory_file, 'structure')
        
        logging.info("Setting up visualization")
        cmd.hide('everything', 'structure')
        cmd.show('cartoon', 'structure')
        cmd.show('lines', 'solvent')
        cmd.color('blue', 'structure')

        logging.info("Calculating RMSD")
        rmsd_file = os.path.join(output_dir, 'rmsd.txt')
        with open(rmsd_file, 'w') as f:
            for state in range(1, cmd.count_states()):
                try:
                    rmsd = cmd.rms_cur(f'structure and name CA and state {state+1}', 'structure and name CA and state 1')
                    f.write(f"State {state+1}: RMSD = {rmsd}\n")
                except Exception as e:
                    logging.error(f"Error calculating RMSD for state {state+1}: {str(e)}")

        logging.info("Generating image")
        cmd.ray(1024, 768)
        image_file = os.path.join(output_dir, 'structure_image.png')
        cmd.png(image_file, dpi=300)

        logging.info("Saving session")
        cmd.save(os.path.join(output_dir, 'analysis_session.pse'))

    except Exception as e:
        logging.error(f"Error in PyMOL analysis: {str(e)}")
    finally:
        logging.info("Closing PyMOL session")
        cmd.quit()

    logging.info("PyMOL analysis completed")
    
# VHT Screening Function
def run_docking(ligand_pdb, receptor_pdb, output_dir):
    autodock_vina_path = "path/to/autodock_vina"  # Update this path
    output_pdbqt = os.path.join(output_dir, f'docked_{os.path.basename(ligand_pdb)}.pdbqt')
    log_file = os.path.join(output_dir, f'log_{os.path.basename(ligand_pdb)}.txt')

    cmd = [
        autodock_vina_path,
        "--receptor", receptor_pdb,
        "--ligand", ligand_pdb,
        "--out", output_pdbqt,
        "--log", log_file
    ]
    run(cmd, check=True)
    logger.info(f"Docking completed for {ligand_pdb}")

def high_throughput_screening(receptor_pdb, ligands_dir, output_dir):
    ligands = [os.path.join(ligands_dir, f) for f in os.listdir(ligands_dir) if f.endswith('.pdbqt')]
    with multiprocessing.Pool() as pool:
        pool.starmap(run_docking, [(ligand, receptor_pdb, output_dir) for ligand in ligands])

# Placeholder VHT Screening Function
def placeholder_vht_screening(receptor_pdb, ligands_dir, output_dir):
    ligands = [f for f in os.listdir(ligands_dir) if f.endswith('.pdbqt')]
    for ligand in ligands:
        # Simulate docking score
        simulated_score = random.uniform(-10.0, 0.0)
        output_file = os.path.join(output_dir, f'docked_{ligand}_result.txt')
        with open(output_file, 'w') as f:
            f.write(f"Simulated docking score for {ligand}: {simulated_score:.2f}")
    logger.info(f"Placeholder VHT screening completed for {len(ligands)} ligands")

def run_Phase_3(protein_results, ligand_results, input_text, output_dir):
    """
    Run Phase 3 analysis including simulation, PyMOL analysis, digital twin simulation, and VHT screening.
    """
    try:
        logger.info("Starting Phase 3 analysis")
        # Ensure the output directory is set to AutoMol-v2/results/phase3
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "phase3")
        os.makedirs(output_dir, exist_ok=True)
        all_analysis_results = []

        for idx, protein_result in enumerate(protein_results):
            protein_output_dir = os.path.join(output_dir, f'protein_{idx}')
            os.makedirs(protein_output_dir, exist_ok=True)

            # Step 1: Run Simulation
            simulation_dir = os.path.join(protein_output_dir, 'simulation')
            os.makedirs(simulation_dir, exist_ok=True)
            simulation_result = run_simulation_pipeline(protein_result['pdb_file'], simulation_dir)

            # Step 2: Run Analysis
            analysis_dir = os.path.join(protein_output_dir, 'analysis')
            os.makedirs(analysis_dir, exist_ok=True)
            analysis_result = run_analysis_pipeline(
                simulation_result['trajectory_file'],
                simulation_result['final_pdb'],
                analysis_dir
            )

            if analysis_result:
                analysis_result['protein_id'] = idx
                analysis_result['simulation_info'] = simulation_result
                all_analysis_results.append(analysis_result)
                logger.info(f"Analysis completed for protein {idx}")
            else:
                logger.warning(f"Analysis failed for protein {idx}")

        # Step 3: PyMOL Analysis
        pymol_output_dir = os.path.join(output_dir, 'pymol_analysis')
        os.makedirs(pymol_output_dir, exist_ok=True)
        for idx, result in enumerate(all_analysis_results):
            structure_file = result['simulation_info']['final_pdb']
            trajectory_file = result['simulation_info']['trajectory_file']
            protein_output_dir = os.path.join(pymol_output_dir, f'protein_{idx}')
            os.makedirs(protein_output_dir, exist_ok=True)
            run_pymol_analysis(structure_file, trajectory_file, protein_output_dir)
            logger.info(f"PyMOL analysis completed for protein {idx}")

        # Step 4: Digital Twin Simulation
        dt_output_dir = os.path.join(output_dir, 'digital_twin')
        os.makedirs(dt_output_dir, exist_ok=True)
        json_path = os.path.join(dt_output_dir, 'aging_model.json')
        simulator = DigitalTwinSimulator(json_path)
        
        drug_effects = {'TERT_activation': 1.0}
        gene_knockouts = ['Tp53', 'NF1', 'CBS']
        condition_changes = {'EX_glucose': -10.0}
        
        fva_result, solution = simulator.simulate_cellular_environment(
            drug_effects=drug_effects,
            gene_knockouts=gene_knockouts,
            condition_changes=condition_changes
        )
        logger.info(f"Digital twin simulation completed. Growth rate: {solution.objective_value}")
        fva_result.to_csv(os.path.join(dt_output_dir, 'fva_results.csv'))

        # Step 5: Virtual High-Throughput Screening
        vht_output_dir = os.path.join(output_dir, 'vht_screening')
        os.makedirs(vht_output_dir, exist_ok=True)
        receptor_pdb = protein_results[0]['pdb_file']  # Assuming the first protein is the receptor
        ligands_dir = os.path.join(output_dir, 'ligands')
        if not os.path.exists(ligands_dir):
            os.makedirs(ligands_dir)
            logger.warning(f"Created ligands directory: {ligands_dir}")
        # For actual docking, use high_throughput_screening
        # high_throughput_screening(receptor_pdb, ligands_dir, vht_output_dir)
        placeholder_vht_screening(receptor_pdb, ligands_dir, vht_output_dir)
        logger.info("Placeholder virtual high-throughput screening completed")

        logger.info("Phase 3 analysis completed successfully")

        phase_3_results = {
            "protein_analyses": all_analysis_results,
            "pymol_analysis_dir": pymol_output_dir,
            "digital_twin_growth_rate": solution.objective_value,
            "fva_results_file": os.path.join(dt_output_dir, 'fva_results.csv'),
            "vht_screening_dir": vht_output_dir
        }
        return phase_3_results

    except Exception as e:
        logger.error(f"Error in Phase 3: {str(e)}")
        raise