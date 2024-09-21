import openmm as mm
import openmm.app as app
import openmm.unit as unit
from sys import stdout
import argparse
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile

def prepare_pdb(input_pdb):
    """Prepare the PDB file by adding missing atoms and removing heterogens."""
    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    return fixer

def generate_trajectory(input_pdb, output_trajectory, output_pdb, simulation_steps=10000):
    """
    Generate a trajectory file from a PDB file using OpenMM.
    """
    print(f"Preparing PDB file: {input_pdb}")
    try:
        fixer = prepare_pdb(input_pdb)
        pdb = app.PDBFile(PDBFile.writeFile(fixer.topology, fixer.positions, open('prepared.pdb', 'w')))
        print("PDB file prepared successfully.")
    except Exception as e:
        print(f"Error preparing PDB file: {str(e)}")
        return

    try:
        print("Creating system...")
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, 
                                         nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)
    except ValueError as e:
        print(f"Error creating system: {str(e)}")
        print("This might be due to non-standard residues or missing atoms.")
        return

    try:
        print("Setting up simulation...")
        integrator = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        print("Minimizing energy...")
        simulation.minimizeEnergy()

        print("Equilibrating...")
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        simulation.step(1000)

        print(f"Running production simulation for {simulation_steps} steps...")
        simulation.reporters.append(app.DCDReporter(output_trajectory, 100))
        simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True, 
            potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
            speed=True, totalSteps=simulation_steps, separator='\t'))

        simulation.step(simulation_steps)

        positions = simulation.context.getState(getPositions=True).getPositions()
        app.PDBFile.writeFile(simulation.topology, positions, open(output_pdb, 'w'))
        print(f"Final state saved to {output_pdb}")
        print(f"Trajectory saved to {output_trajectory}")
    except Exception as e:
        print(f"Error during simulation: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a trajectory file from a PDB file.")
    parser.add_argument("input_pdb", help="Path to the input PDB file")
    parser.add_argument("output_trajectory", help="Path to save the output trajectory (DCD) file")
    parser.add_argument("output_pdb", help="Path to save the final PDB file")
    parser.add_argument("--steps", type=int, default=10000, help="Number of simulation steps (default: 10000)")
    args = parser.parse_args()

    generate_trajectory(args.input_pdb, args.output_trajectory, args.output_pdb, args.steps)