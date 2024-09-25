import os
import logging
from typing import List, Dict, Any
from openmm import LangevinMiddleIntegrator, Vec3
from openmm.app import (
    PDBFile,
    Simulation,
    PDBReporter,
    StateDataReporter,
    PME,
    HBonds,
    ForceField,
    Modeller
)
from openmm.unit import nanometer, kelvin, picosecond, picoseconds
from pdbfixer import PDBFixer
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
from math import floor

# Configure the logger
logger = logging.getLogger(__name__)

def calculate_rmsd(structure_file, trajectory_file):
    # Load the trajectory
    u = mda.Universe(structure_file, trajectory_file)

    # Calculate RMSD
    R = rms.RMSD(u, u, select="backbone", ref_frame=0)
    R.run()

    # Plot RMSD
    plt.figure(figsize=(10, 6))
    plt.plot(R.results.time, R.results.rmsd)
    plt.xlabel('Time (ps)')
    plt.ylabel('RMSD (Å)')
    plt.title('RMSD over time')
    plt.savefig('rmsd_plot.png')
    plt.close()

def molecular_dynamics_simulation(pdb_file: str, output_dir: str) -> Dict[str, Any]:
    try:
        fixer = PDBFixer(filename=pdb_file)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)  # pH 7.0

        fixed_pdb = os.path.join(output_dir, 'fixed.pdb')
        with open(fixed_pdb, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        logger.info(f"Fixed PDB written to {fixed_pdb}")

        # Create the system
        # Updated ForceField to use 'tip3p.xml' instead of 'tip3pfb.xml'
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
        pdb = PDBFile(fixed_pdb)
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)

        logger.info("Adding solvent...")
        try:
            modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometer)
        except Exception as e:
            logger.error(f"Error adding solvent: {str(e)}")
            raise

        logger.info("Creating system...")
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1*nanometer,
            constraints=HBonds
        )

        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        # Minimize energy
        logger.info("Minimizing energy...")
        simulation.minimizeEnergy()

        # Equilibrate
        logger.info("Equilibrating...")
        simulation.context.setVelocitiesToTemperature(300*kelvin)
        simulation.step(5000)  # Equilibration steps

        # Production run
        logger.info("Starting production run...")
        trajectory_file = os.path.join(output_dir, 'trajectory.pdb')

        def valid_coords(simulation, state):
            positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            return all(isinstance(coord, (int, float)) for pos in positions for coord in pos)

        # Removed 'checkForErrors=True' as it's not a valid argument
        simulation.reporters.append(PDBReporter(trajectory_file, 1000, enforcePeriodicBox=False))
        simulation.reporters.append(StateDataReporter(
            os.path.join(output_dir, 'output.csv'),
            1000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=20000,
            separator='\t'
        ))

        total_steps = 20000
        step_size = 1000
        for step in range(0, total_steps, step_size):
            current_state = simulation.context.getState(getPositions=True)
            if not valid_coords(simulation, current_state):
                logger.error(f"Invalid coordinates detected at step {step}")
                break
            simulation.step(step_size)
            logger.info(f"Completed {step + step_size} steps out of {total_steps}")

        logger.info("Molecular dynamics simulation completed.")

        simulation_output = {
            "pdb_file": pdb_file,
            "trajectory_file": trajectory_file,
            "other_metrics": {}
        }

        return simulation_output

    except Exception as e:
        logger.error(f"Error during molecular dynamics simulation: {str(e)}")
        raise

def simulate(protein: Dict[str, Any], simulation_dir: str) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    protein_id = protein.get("id")
    logger.debug(f"Protein ID: {protein_id}")
    pdb_file = protein.get("pdb_file")
    logger.debug(f"PDB file: {pdb_file}")
    protein_sim_dir = os.path.join(simulation_dir, f"protein_{protein_id}")
    os.makedirs(protein_sim_dir, exist_ok=True)
    logger.debug(f"Protein simulation directory: {protein_sim_dir}")
    simulation_output = molecular_dynamics_simulation(pdb_file, protein_sim_dir)
    return simulation_output

def run_simulation_pipeline(protein_results: List[Dict[str, Any]], simulation_dir: str, device: str) -> List[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    logger.info(f"Running simulations on device: {device}")

    simulation_results = []
    for protein in protein_results:
        result = simulate(protein, simulation_dir)
        simulation_results.append(result)

    return simulation_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Real input data
    protein_results = [
        {"id": "1", "pdb_file": r"C:\Users\wes\AutoMol-v2\dummy_simulation_output\AF-P02686-F1-model_v4.pdb"},
        {"id": "2", "pdb_file": r"C:\Users\wes\AutoMol-v2\dummy_simulation_output\AF-P07237-F1-model_v4.pdb"}
    ]
    simulation_dir = "simulation_output"
    device = "cpu"

    # Run the simulation pipeline
    results = run_simulation_pipeline(protein_results, simulation_dir, device)
    print("Simulation results:", results)
