from openmm import *
from openmm.app import *
from openmm.unit import *

import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt

from openmm import LangevinMiddleIntegrator, System
from openmm.app import PDBFile, Simulation, PDBReporter, StateDataReporter, PME, HBonds, ForceField
from openmm.unit import nanometer, kelvin, picosecond, picoseconds
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from openmm import *
from openmm.app import *
from openmm.unit import *
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
import matplotlib.pyplot as plt
from openmm.app import PDBFile, Modeller, ForceField

from openmm.app import PDBFile, Simulation, PDBReporter, StateDataReporter, PME, HBonds, ForceField, NoCutoff
from openmm import LangevinMiddleIntegrator, System
from openmm.unit import nanometer, kelvin, picosecond, picoseconds, angstrom
from openmm.vec3 import Vec3


def molecular_dynamics_simulation(pdb_file):
    # Fix the PDB file
    print("Fixing PDB file...")
    fixer = PDBFixer(pdb_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # Add hydrogens appropriate for pH 7.0

    # Create a PDBFile object from the fixed structure
    PDBFile.writeFile(fixer.topology, fixer.positions, open('fixed.pdb', 'w'))
    pdb = PDBFile('fixed.pdb')

    # Create the system
    print("Creating system...")
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)  # This adds missing hydrogen atoms
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometer)

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, 
                                    nonbondedCutoff=1*nanometer, constraints=HBonds)

    # Create the integrator
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)

    # Create the simulation
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # Minimize and equilibrate
    print("Minimizing energy...")
    simulation.minimizeEnergy()
    
    print("Setting velocities...")
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    
    print("Equilibrating...")
    simulation.step(10000)  # equilibration

    # Production run
    logger.info("Starting production run...")
    trajectory_file = os.path.join(output_dir, 'trajectory.pdb')
    
    def valid_coords(simulation, state):
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        return all(isinstance(coord, (int, float)) for pos in positions for coord in pos)

    simulation.reporters.append(PDBReporter(trajectory_file, 1000, enforcePeriodicBox=False, checkForErrors=True))
    simulation.reporters.append(StateDataReporter(
        os.path.join(output_dir, 'output.csv'),
        1000,
        step=True,
        potentialEnergy=True,
        temperature=True,
        progress=True,
        remainingTime=True,
        speed=True,
        totalSteps=num_steps,
        separator='\t'))

    try:
        for step in range(0, num_steps, step_size):
            if not valid_coords(simulation, simulation.context.getState(getPositions=True)):
                logger.error(f"Invalid coordinates detected at step {step}")
                break
            simulation.step(step_size)
    except Exception as e:
        logger.error(f"Error during molecular dynamics simulation: {str(e)}")
        raise


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
    
    
if __name__ == "__main__":
    pdb_file = "./em3_predictions/generation.pdb"
    
    # Load the PDB file
    pdb = PDBFile(pdb_file)
    
    # Create the forcefield
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    
    # Create the modeller
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)  # This adds missing hydrogen atoms
    
    # Run MD simulation
    molecular_dynamics_simulation(pdb_file)

    # Calculate and plot RMSD
    calculate_rmsd(pdb_file, "trajectory.pdb")

    print("Molecular dynamics simulation and RMSD calculation completed.")
