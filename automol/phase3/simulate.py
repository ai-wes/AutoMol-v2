import os
import logging
import mdtraj as md
import tempfile
import numpy as np
from openmm import *
from openmm.app import *
from openmm import unit
from pdbfixer import PDBFixer
import nglview as nv
import tempfile
from openmm import Platform, LangevinMiddleIntegrator, OpenMMException
from openmm.app import PDBFile, ForceField, Modeller, Simulation, DCDReporter, StateDataReporter
from openmm.unit import nanometer, kelvin, picosecond, femtosecond
from openmm.app import PME, HBonds
from openmm.app import PDBFile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_system(pdb_file):
    """Prepare the system for simulation."""
    try:
        # Use PDBFixer to add missing atoms and hydrogens
        fixer = PDBFixer(filename=pdb_file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        # Save the fixed structure to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".pdb", delete=False) as temp_file:
            PDBFile.writeFile(fixer.topology, fixer.positions, temp_file)
            tmp_filename = temp_file.name

        # Now use the fixed PDB file to create the system
        pdb = PDBFile(tmp_filename)
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        modeller = Modeller(pdb.topology, pdb.positions)

        # Center the protein in the box

        # Center the protein in the box
        protein_atoms = [atom.index for atom in modeller.topology.atoms() if atom.residue.name != 'HOH']
        protein_positions = [modeller.positions[i] for i in protein_atoms]
        
        # Convert Quantity objects to numpy arrays for calculation
        protein_positions_np = np.array([p.value_in_unit(unit.nanometer) for p in protein_positions])
        center = np.mean(protein_positions_np, axis=0) * unit.nanometer

        for i in range(len(modeller.positions)):
            modeller.positions[i] = modeller.positions[i] - center

        # Add solvent
        modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * unit.nanometer, neutralize=True)

        system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME,
                                         nonbondedCutoff=1.0 * unit.nanometer,
                                         constraints=HBonds)
        logger.info(f"System prepared for: {pdb_file}")

        # Remove the temporary file
        os.unlink(tmp_filename)

        return system, modeller
    except Exception as e:
        logger.error(f"Error preparing system for {pdb_file}: {str(e)}")
        raise

def run_simulation(system, modeller, output_dir, steps=10000):
    try:
        integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 1.0*unit.femtosecond)
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}

        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        simulation.context.setPositions(modeller.positions)

        logger.info("Minimizing energy...")
        simulation.minimizeEnergy(maxIterations=5000)

        logger.info("Heating and equilibrating...")
        for temp in range(0, 301, 50):
            integrator.setTemperature(temp*kelvin)
            simulation.step(1000)
            
        simulation.context.setVelocitiesToTemperature(300*kelvin)

        logger.info(f"Running production simulation for {steps} steps...")
        simulation.reporters.append(DCDReporter(os.path.join(output_dir, 'trajectory.dcd'), 1000))
        simulation.reporters.append(StateDataReporter(os.path.join(output_dir, 'output.txt'), 1000, 
            step=True, potentialEnergy=True, temperature=True, progress=True, 
            remainingTime=True, speed=True, totalSteps=steps, separator='\t'))

        for i in range(0, steps, 1000):
            try:
                simulation.step(1000)
                logger.info(f"Simulation progress: {i+1000}/{steps} steps ({(i+1000)/500:.1f} ps)")
            except OpenMMException as e:
                logger.warning(f"OpenMM exception at step {i}: {str(e)}. Attempting to continue...")
                simulation.context.setVelocitiesToTemperature(300*kelvin)

        positions = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(simulation.topology, positions, open(os.path.join(output_dir, 'final.pdb'), 'w'))

        logger.info("Simulation completed successfully.")
        return os.path.join(output_dir, 'trajectory.dcd'), os.path.join(output_dir, 'final.pdb')
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        raise

def check_steric_clashes(pdb_file, cutoff=0.4*unit.nanometer):
    structure = PDBFile(pdb_file)
    positions = structure.getPositions()
    num_atoms = len(positions)
    
    clash_count = 0
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            distance = unit.norm(positions[i] - positions[j])
            if distance < cutoff:
                logger.warning(f"Steric clash detected between atoms {i} and {j}: distance = {distance}")
                clash_count += 1

    logger.info(f"Steric clash check completed. Total clashes detected: {clash_count}")
    return clash_count


def visualize_trajectory(trajectory_file, pdb_file, output_file):
    """Create an HTML visualization of the trajectory."""
    try:
        traj = md.load(trajectory_file, top=pdb_file)
        view = nv.show_mdtraj(traj)
        view.render_image()
        view.download_image(output_file)
        logger.info(f"Trajectory visualization saved to {output_file}")
    except Exception as e:
        logger.error(f"Error creating trajectory visualization: {str(e)}")
        logger.info("Skipping visualization due to error.")
        return None  # Return None instead of raising an exception

def run_simulation_pipeline(pdb_file, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Preparing system for {pdb_file}")

        system, modeller = prepare_system(pdb_file)

        logger.info(f"Running simulation for {pdb_file}")
        trajectory_file, final_pdb = run_simulation(system, modeller, output_dir)
        check_steric_clashes(final_pdb)

        # Create visualization
        visualization_file = os.path.join(output_dir, 'trajectory_visualization.png')
        visualize_trajectory(trajectory_file, final_pdb, visualization_file)

        logger.info(f"Simulation results saved in: {output_dir}")

        return {
            "input_pdb": pdb_file,
            "output_dir": output_dir,
            "trajectory_file": trajectory_file,
            "final_pdb": final_pdb,
            "visualization_file": visualization_file
        }
    except Exception as e:
        logger.error(f"Error in simulation pipeline for {pdb_file}: {str(e)}")
        raise