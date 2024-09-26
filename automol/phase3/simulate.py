from openmm import Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.app import PDBFile, PDBReporter, StateDataReporter, Modeller, ForceField, Simulation, HBonds, PME
from openmm.unit import nanometer, kelvin, picosecond, picoseconds, bar
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, Any, List
from server.app import emit_progress


# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def molecular_dynamics_simulation(
    pdb_file: str, output_dir: str, total_steps: int = 20000, step_size: int = 1000
) -> Dict[str, Any]:
    """
    Perform molecular dynamics simulation on a given PDB file.
    """
    try:
        logger.info(f"Loading PDB file: {pdb_file}")
        emit_progress(f"Loading PDB file: {pdb_file}")
        pdb = PDBFile(pdb_file)

        logger.info("Defining forcefield")
        emit_progress("Defining forcefield")
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        logger.info("Cleaning up and adding hydrogens")
        emit_progress("Cleaning up and adding hydrogens")
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.deleteWater()
        modeller.addHydrogens(forcefield)

        logger.info("Adding solvent")
        emit_progress("Adding solvent")
        modeller.addSolvent(forcefield, padding=1.0*nanometer)

        logger.info("Setting up system and integrator")
        emit_progress("Setting up system and integrator")
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, 
                                         nonbondedCutoff=1.0*nanometer, constraints=HBonds)
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)

        # Set up GPU acceleration
        platform = Platform.getPlatformByName('CUDA')
        properties = {'DeviceIndex': '0,1', 'Precision': 'mixed'}
        
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        simulation.context.setPositions(modeller.positions)

        logger.info("Minimizing energy")
        emit_progress("Minimizing energy")
        simulation.minimizeEnergy()

        trajectory_file = os.path.join(output_dir, "trajectory.pdb")
        log_file = os.path.join(output_dir, "md_log.txt")

        simulation.reporters.append(PDBReporter(trajectory_file, step_size))
        simulation.reporters.append(StateDataReporter(log_file, step_size, step=True,
            potentialEnergy=True, temperature=True, volume=True))
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy()  # Perform an energy minimization before the main simulation

        logger.info("Running NVT equilibration")
        emit_progress("Running NVT equilibration")
        simulation.step(10000)  # 10,000 steps of NVT equilibration

        logger.info("Adding barostat for NPT simulation")
        emit_progress("Adding barostat for NPT simulation")
        system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
        simulation.context.reinitialize(preserveState=True)

        logger.info("Running NPT production MD")
        emit_progress("Running NPT production MD")
        simulation.step(total_steps)

        logger.info("Simulation completed. Performing basic analysis.")
        emit_progress("Simulation completed. Performing basic analysis.")
        data = np.loadtxt(log_file, delimiter=',', skiprows=1)
        
        step = data[:,0]
        potential_energy = data[:,1]
        temperature = data[:,2]
        volume = data[:,3]

        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.plot(step, potential_energy)
        plt.xlabel("Step")
        plt.ylabel("Potential energy (kJ/mol)")
        
        plt.subplot(3, 1, 2)
        plt.plot(step, temperature)
        plt.xlabel("Step")
        plt.ylabel("Temperature (K)")
        
        plt.subplot(3, 1, 3)
        plt.plot(step, volume)
        plt.xlabel("Step")
        plt.ylabel("Volume (nm^3)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "analysis_plots.png"))
        plt.close()

        return {
            "pdb_file": pdb_file,
            "trajectory_file": trajectory_file,
            "log_file": log_file,
            "analysis_plot": os.path.join(output_dir, "analysis_plots.png"),
        }
    
    except Exception as e:
        logger.error(f"An error occurred during simulation: {str(e)}")
        emit_progress(f"An error occurred during simulation: {str(e)}")
        # Print more detailed error information
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}




def simulate(protein: Dict[str, Any], simulation_dir: str) -> Dict[str, Any]:
    """
    Simulate a single protein.

    Parameters:
    - protein: Dictionary containing protein information with 'id' and 'pdb_file'.
    - simulation_dir: Base directory for simulations.

    Returns:
    A dictionary with simulation result file paths.
    """
    protein_id = protein["id"]
    pdb_file = protein["pdb_file"]
    protein_sim_dir = os.path.join(simulation_dir, f"protein_{protein_id}")
    os.makedirs(protein_sim_dir, exist_ok=True)
    emit_progress(f"Starting simulation for protein {protein_id}")
    return molecular_dynamics_simulation(pdb_file, protein_sim_dir)

def run_simulation_pipeline(protein_results: List[Dict[str, Any]], simulation_dir: str) -> List[Dict[str, Any]]:
    """
    Run simulations for a list of proteins.

    Parameters:
    - protein_results: List of dictionaries with protein information.
    - simulation_dir: Base directory for all simulations.

    Returns:
    List of dictionaries containing simulation result file paths for each protein.
    """
    emit_progress("Starting simulation pipeline")
    results = []
    for protein in protein_results:
        emit_progress(f"Simulating protein {protein['id']}")
        results.append(simulate(protein, simulation_dir))
    emit_progress("Simulation pipeline completed")
    return results

if __name__ == "__main__":
    protein_results = [
        {"id": "1", "pdb_file": r"C:\Users\wes\AutoMol-v2\dummy_simulation_output\AF-P02686-F1-model_v4.pdb"},
        {"id": "2", "pdb_file": r"C:\Users\wes\AutoMol-v2\dummy_simulation_output\AF-P07237-F1-model_v4.pdb"}
    ]
    simulation_dir = "simulation_output"
    os.makedirs(simulation_dir, exist_ok=True)

    results = run_simulation_pipeline(protein_results, simulation_dir)
    print("Simulation results:", results)