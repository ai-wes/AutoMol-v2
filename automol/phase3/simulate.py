from multiprocessing import Pool, cpu_count
# AutoMol-v2/automol/phase3/simulate.py

import os
import logging
from openmm import LangevinMiddleIntegrator, System
from openmm.app import PDBFile, Simulation, PDBReporter, StateDataReporter, PME, HBonds, ForceField, Modeller
from openmm.unit import nanometer, kelvin, picosecond, picoseconds
from pdbfixer import PDBFixer
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def molecular_dynamics_simulation(pdb_file, simulation_dir):
    """
    Perform molecular dynamics simulation on the given PDB file.
    
    Parameters:
    - pdb_file: Path to the input PDB file.
    - simulation_dir: Directory to store simulation outputs.
    
    Returns:
    - Dictionary containing simulation output paths.
    """
    try:
        logger.info(f"Starting MD simulation for {pdb_file}")
        
        # Fix the PDB file
        fixer = PDBFixer(filename=pdb_file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)  # pH 7.0
        
        fixed_pdb = os.path.join(simulation_dir, 'fixed.pdb')
        with open(fixed_pdb, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        logger.info(f"Fixed PDB written to {fixed_pdb}")
        
        # Create the system
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        pdb = PDBFile(fixed_pdb)
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)
        modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometer)
        
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
        simulation.step(10000)  # Equilibration steps
        
        # Production run
        logger.info("Starting production run...")
        trajectory_file = os.path.join(simulation_dir, 'trajectory.pdb')
        simulation.reporters.append(PDBReporter(trajectory_file, 1000))
        simulation.reporters.append(StateDataReporter(
            os.path.join(simulation_dir, 'output.csv'),
            1000, step=True, potentialEnergy=True, temperature=True
        ))
        simulation.step(1000000)  # 1 ns simulation
        
        logger.info("Molecular dynamics simulation completed.")
        
        return {
            "fixed_pdb": fixed_pdb,
            "trajectory_file": trajectory_file,
            "output_csv": os.path.join(simulation_dir, 'output.csv')
        }
    
    except Exception as e:
        logger.error(f"Error during MD simulation: {str(e)}", exc_info=True)
        raise


def run_simulation_pipeline(protein_results, simulation_dir, device):


    
    logger.info(f"Running simulations on device: {device}")
    
    with Pool(processes=min(cpu_count(), len(protein_results))) as pool:
        simulation_results = pool.map(simulate, protein_results)
    
    logger.info(f"Completed all simulations.")
    return simulation_results


def simulate(protein):
    protein_id = protein.get("id")
    pdb_file = protein.get("pdb_file")
    protein_sim_dir = os.path.join(simulation_dir, f"protein_{protein_id}")
    os.makedirs(protein_sim_dir, exist_ok=True)
    return molecular_dynamics_simulation(pdb_file, protein_sim_dir)
