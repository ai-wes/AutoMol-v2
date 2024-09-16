import asyncio
import logging
import os
from typing import List, Dict, Any
import sys
import tenacity

from pathlib import Path
            
            
            
from deap import base, creator, tools, algorithms
from rdkit.Chem import Descriptors, AllChem
import numpy as np
import logging
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from deap import base, creator, tools, algorithms
import mdtraj as md
from openmm import app, unit
from openmm.app import PDBFile, Modeller, Simulation
from openmm import LangevinIntegrator, Platform
import random
from rdkit.Chem import AllChem, Descriptors
from deap import base, creator, tools, algorithms

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rdkit import Chem
import logging
import asyncio
from .optimize_ligand import optimize_ligand_smiles
from ...utils.shared_state import get_protein_sequences
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Checkpoint decorator
def checkpoint(step_name):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            logger.info(f"Starting step: {step_name}")
            try:
                result = await func(*args, **kwargs)
                logger.info(f"Completed step: {step_name}")
                return result
            except Exception as e:
                logger.error(f"Error in step {step_name}: {str(e)}")
                raise
        return wrapper
    return decorator


def validate_smiles(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Check for valid atom types
        valid_atoms = set(['C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'])
        if not all(atom.GetSymbol() in valid_atoms for atom in mol.GetAtoms()):
            return False
        
        # Check molecular weight
        mol_weight = Descriptors.ExactMolWt(mol)
        if mol_weight < 100 or mol_weight > 1000:
            return False
        
        # Check number of rotatable bonds
        n_rotatable = AllChem.CalcNumRotatableBonds(mol)
        if n_rotatable > 10:
            return False
        
        # Check for kekulization
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except:
            return False
        
        return True
    except Exception as e:
        logger.error(f"Validation error for SMILES {smiles}: {e}")
        return False    



class SMILESGenerator:
    def __init__(self, model_name: str = "laituan245/t5-v1_1-large-caption2smiles"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
    
    async def load_model(self):
        if self.tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, model_max_length=1024)
        if self.model is None:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode
            if torch.cuda.is_available():
                self.model = self.model.cuda()
    
    async def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Model unloaded and CUDA cache cleared.")
    
# Phase_2/SMILESLigandPipeline.py




    @checkpoint("SMILES Generation")
    async def generate(self, technical_instruction: str, max_attempts: int = 10) -> str:
        await self.load_model()
        try:
            for attempt in range(max_attempts):
                try:
                    input_ids = self.tokenizer(technical_instruction, return_tensors="pt").input_ids
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                    outputs = self.model.generate(input_ids, num_beams=5, max_length=1024)
                    smiles = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info(f"Generated SMILES: {smiles}")

                    # Get the protein sequences
                    protein_sequences = await get_protein_sequences()
                    
                    # Use the first protein sequence for optimization
                    # You might want to modify this if you need to use multiple sequences
                    protein_sequence = protein_sequences[0] if protein_sequences else ""

                    optimized_smiles = await optimize_ligand_smiles(smiles, protein_sequence)

                    if validate_smiles(optimized_smiles):
                        return optimized_smiles
                    else:
                        raise ValueError("Generated an invalid SMILES string.")
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_attempts - 1:
                        logger.error(f"Failed to generate valid SMILES after {max_attempts} attempts.")
                        raise
                    await asyncio.sleep(1)  # Wait a bit before retrying
        finally:
            await self.unload_model()            
            
            
# Phase_2/SMILESLigandPipeline.py


class SMILESOptimizer:
    def __init__(self, population_size=50, generations=20):
        self.population_size = population_size
        self.generations = generations

        # Define a single-objective fitness function to maximize LogP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.mate_molecules)
        self.toolbox.register("mutate", self.mutate_molecule)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.fitness_function)

    def init_individual(self):
        # Start with the original SMILES
        return creator.Individual([self.original_smiles])

    @checkpoint("SMILES Optimization")
    async def optimize(self, smiles: str) -> str:
        self.original_smiles = smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES provided for optimization.")

        population = self.toolbox.population(n=self.population_size)

        # Evaluate the entire population
        for individual in population:
            individual.fitness.values = self.toolbox.evaluate(individual)

        # Begin the evolution
        for gen in range(self.generations):
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.5, mutpb=0.2)
            fits = list(map(self.toolbox.evaluate, offspring))

            # Assign fitness
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            # Select the next generation population
            population = self.toolbox.select(offspring, k=len(population))

            logger.info(f"Generation {gen + 1} complete.")

            # Early stopping if no improvement
            if all(ind.fitness.values[0] <= 0 for ind in population):
                logger.warning("All individuals have non-positive fitness. Stopping early.")
                break

        # Select the best individual
        best_ind = tools.selBest(population, k=1)[0]
        optimized_smiles = best_ind[0]
        logger.info(f"Optimized SMILES: {optimized_smiles}")
        return optimized_smiles

    def fitness_function(self, individual):
        smiles = individual[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (-1.0,)  # Penalize invalid molecules

        log_p = Descriptors.MolLogP(mol)
        qed = Descriptors.qed(mol)

        # Combine LogP and QED for fitness
        fitness = log_p + qed
        if fitness <= 0:
            return (-1.0,)
        return (fitness,)

    def mate_molecules(self, ind1, ind2):
        # Simple crossover: swap SMILES strings
        ind1[0], ind2[0] = ind2[0], ind1[0]
        return ind1, ind2

    def mutate_molecule(self, individual):
        smiles = individual[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return individual,

        # Randomly add or remove an atom
        if random.random() < 0.5 and mol.GetNumAtoms() > 1:
            # Remove a random atom
            atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
            try:
                mol = Chem.RWMol(mol)
                mol.RemoveAtom(atom_idx)
                new_smiles = Chem.MolToSmiles(mol)
                individual[0] = new_smiles
            except:
                pass
        else:
            # Add a random atom
            atom_symbol = random.choice(['C', 'N', 'O', 'F', 'Cl', 'Br'])
            mol = Chem.RWMol(mol)
            new_atom = Chem.Atom(atom_symbol)
            mol.AddAtom(new_atom)
            # Connect the new atom to a random existing atom
            if mol.GetNumAtoms() > 1:
                bond_idx = random.randint(0, mol.GetNumAtoms() - 2)
                mol.AddBond(mol.GetNumAtoms() - 1, bond_idx, Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(mol)
                new_smiles = Chem.MolToSmiles(mol)
                individual[0] = new_smiles
            except:
                pass
        return (individual,)

logger = logging.getLogger(__name__)
class StructurePredictor:
    @checkpoint("3D Structure Prediction")
    async def predict(self, smiles: str, output_dir: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string for structure prediction.")

            mol = Chem.AddHs(mol)
            num_confs = 50
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=num_confs,
                params=AllChem.ETKDGv3(),
                pruneRmsThresh=0.5
            )

            if not conformer_ids:
                raise ValueError("Embedding failed for the molecule.")

            energies = []
            for conf_id in conformer_ids:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
                    energy = ff.CalcEnergy()
                    energies.append((energy, conf_id))
                except:
                    continue

            if not energies:
                raise ValueError("MMFF optimization failed for all conformers.")

            # Select the conformer with the lowest energy
            min_energy_conf = min(energies, key=lambda x: x[0])[1]
            mol.SetProp("_conformerId", str(min_energy_conf))

            ligand_pdb_path = os.path.join(output_dir, f"{smiles[:10]}_ligand.pdb")
            Chem.MolToPDBFile(mol, ligand_pdb_path, confId=min_energy_conf)
            logger.info(f"3D structure saved to: {ligand_pdb_path}")
            return ligand_pdb_path
        except Exception as e:
            logger.error(f"Error in 3D structure prediction: {str(e)}")
            raise
        
        
        
from Bio.PDB import PDBParser

# 4. Ensemble Docking
class EnsembleDocking:
    def __init__(self, vina_path: str = "vina", exhaustiveness: int = 8, num_modes: int = 9):
        self.vina_path = vina_path
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes

    def get_binding_site_center(self, protein_pdb: str, reference_ligand: str = None) -> tuple:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', protein_pdb)
        model = structure[0]
        # Implement logic to determine binding site center
        # For example, find residues within X angstroms of the reference ligand
        # Placeholder coordinates:
        return (0.0, 0.0, 0.0)

    @checkpoint("Ensemble Docking")
    async def dock_ensemble(self, ligand_path: str, protein_ensemble: List[str], output_dir: str) -> List[Dict[str, Any]]:
        docking_tasks = [
            self.dock_single(ligand_path, protein, output_dir, i)
            for i, protein in enumerate(protein_ensemble)
        ]
        return await asyncio.gather(*docking_tasks, return_exceptions=True)

    async def dock_single(self, ligand_path: str, protein_path: str, output_dir: str, index: int) -> Dict[str, Any]:
        output_path = os.path.join(output_dir, f"docked_{index}.pdbqt")
        log_path = os.path.join(output_dir, f"docking_log_{index}.txt")

        # Prepare receptor and ligand
        receptor_pdbqt = await self._prepare_pdbqt(protein_path, is_receptor=True)
        ligand_pdbqt = await self._prepare_pdbqt(ligand_path, is_receptor=False)

        # Get binding site center dynamically
        center_x, center_y, center_z = self.get_binding_site_center(protein_path)
        size_x, size_y, size_z = 20.0, 20.0, 20.0  # You might want to adjust these based on the binding site

        cmd = (
            f"{self.vina_path} --receptor {receptor_pdbqt} --ligand {ligand_pdbqt} "
            f"--center_x {center_x} --center_y {center_y} --center_z {center_z} "
            f"--size_x {size_x} --size_y {size_y} --size_z {size_z} "
            f"--out {output_path} --log {log_path} "
            f"--exhaustiveness {self.exhaustiveness} --num_modes {self.num_modes}"
        )

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Docking failed for protein {index}: {stderr.decode()}")
            return {"index": index, "affinity": float('inf')}

        # Parse the log file to get the best affinity
        best_affinity = float('inf')
        try:
            with open(log_path, 'r') as log_file:
                for line in log_file:
                    if line.strip().startswith('-----+------------+----------+----------'):
                        next(log_file)  # Skip header line
                        affinity_line = next(log_file)
                        best_affinity = float(affinity_line.split()[1])
                        break
        except Exception as e:
            logger.error(f"Failed to parse docking log for protein {index}: {e}")
            best_affinity = float('inf')

        return {"index": index, "affinity": best_affinity, "docked_ligand": output_path}

    async def _prepare_pdbqt(self, pdb_path: str, is_receptor: bool) -> str:
        pdbqt_path = pdb_path.replace('.pdb', '.pdbqt')
        if os.path.exists(pdbqt_path):
            return pdbqt_path

        if is_receptor:
            prepare_command = f"prepare_receptor -r {pdb_path} -o {pdbqt_path}"
        else:
            prepare_command = f"prepare_ligand -l {pdb_path} -o {pdbqt_path}"

        process = await asyncio.create_subprocess_shell(
            prepare_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Failed to prepare {'receptor' if is_receptor else 'ligand'}: {stderr.decode()}")
        return pdbqt_path

# 5. Docking Simulation
class DockingSimulator:
    @checkpoint("Docking Simulation")
    async def dock(self, protein_pdbqt: str, ligand_pdbqt: str, output_dir: str) -> str:
        try:
            output_path = os.path.join(output_dir, "docked.pdbqt")
            log_path = os.path.join(output_dir, "docking_log.txt")

            # Define docking box parameters (replace with appropriate values)
            center_x, center_y, center_z = 0.0, 0.0, 0.0
            size_x, size_y, size_z = 20.0, 20.0, 20.0

            docking_command = (
                f"vina --receptor {protein_pdbqt} --ligand {ligand_pdbqt} "
                f"--center_x {center_x} --center_y {center_y} --center_z {center_z} "
                f"--size_x {size_x} --size_y {size_y} --size_z {size_z} "
                f"--out {output_path} --log {log_path}"
            )

            process = await asyncio.create_subprocess_shell(
                docking_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Docking failed: {stderr.decode()}")

            logger.info(f"Docking completed. Output saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error in docking simulation: {str(e)}")
            raise

# 6. Analysis
class DockingAnalyzer:
    @checkpoint("Docking Analysis")
    async def analyze(self, docked_ligand: str, protein_pdb: str, simulation_output_dir: str) -> Dict[str, Any]:
        try:
            # Run MD simulation
            simulation_output = await self._run_md_simulation(docked_ligand, protein_pdb, simulation_output_dir)

            # Analyze trajectory
            results = await self._analyze_trajectory(simulation_output, protein_pdb)

            logger.info(f"Analysis completed. Results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error in docking analysis: {str(e)}")
            raise

    async def _run_md_simulation(self, docked_ligand: str, protein_pdb: str, output_dir: str) -> str:
        pdb = PDBFile(protein_pdb)
        ligand = PDBFile(docked_ligand)

        # Combine protein and ligand
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.add(ligand.topology, ligand.positions)

        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1*unit.nanometer,
                                         constraints=app.HBonds)

        # Solvate the system
        modeller.addSolvent(forcefield, model='tip3p', padding=1.0*unit.nanometer)

        integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond,
                                        0.002*unit.picoseconds)

        platform = Platform.getPlatformByName('CUDA' if Platform.getNumPlatforms() > 1 else 'CPU')
        simulation = Simulation(modeller.topology, system, integrator, platform)
        simulation.context.setPositions(modeller.positions)

        # Energy Minimization
        simulation.minimizeEnergy()

        # Equilibration
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        simulation.step(10000)  # 20 ps equilibration

        # Production run
        output_file = os.path.join(output_dir, "trajectory.dcd")
        simulation.reporters.append(app.DCDReporter(output_file, 1000))
        simulation.reporters.append(app.StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))
        simulation.step(50000)  # 100 ps production

        return output_file

    async def _analyze_trajectory(self, trajectory_file: str, protein_pdb: str) -> Dict[str, float]:
        traj = md.load(trajectory_file, top=protein_pdb)
        rmsd = md.rmsd(traj, traj, 0)
        rmsf = md.rmsf(traj, traj, 0)
        avg_rmsd = np.mean(rmsd)
        max_rmsd = np.max(rmsd)
        avg_rmsf = np.mean(rmsf)
        max_rmsf = np.max(rmsf)

        return {
            "avg_rmsd": avg_rmsd,
            "max_rmsd": max_rmsd,
            "avg_rmsf": avg_rmsf,
            "max_rmsf": max_rmsf,
        }
# 7. Scoring and Selection
class LigandScorer:
    @checkpoint("Ligand Scoring")
    async def score(self, analysis_results: List[Dict[str, Any]], docking_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            best_score = float('-inf')
            best_ligand = None
            for analysis_result, docking_result in zip(analysis_results, docking_results):
                score = self._calculate_score(analysis_result, docking_result)
                if score > best_score:
                    best_score = score
                    best_ligand = {**analysis_result, **docking_result, "score": score}

            if best_ligand is None:
                raise ValueError("No valid ligands found during scoring.")

            logger.info(f"Best ligand selected with score: {best_score}")
            return best_ligand
        except Exception as e:
            logger.error(f"Error in ligand scoring: {str(e)}")
            raise

    def _calculate_score(self, analysis_result: Dict[str, Any], docking_result: Dict[str, Any]) -> float:
        # Combine docking affinity and simulation results
        affinity = docking_result.get('affinity', float('inf'))
        avg_rmsd = analysis_result.get('avg_rmsd', float('inf'))
        max_rmsd = analysis_result.get('max_rmsd', float('inf'))

        # Scoring function (weights can be adjusted)
        score = -affinity - avg_rmsd - 0.5 * max_rmsd
        return score

# 8. Full Pipeline Integration
class SMILESLigandPipeline:
    def __init__(self):
        self.generator = SMILESGenerator()
        self.optimizer = SMILESOptimizer()
        self.predictor = StructurePredictor()
        self.ensemble_docker = EnsembleDocking()
        self.analyzer = DockingAnalyzer()
        self.scorer = LigandScorer()

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def run(self, technical_instruction: str, protein_ensemble: List[str], output_dir: str) -> Dict[str, Any]:
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Starting SMILES Generation")
            smiles = await self.generator.generate(technical_instruction)
            logger.info(f"SMILES Generation completed: {smiles}")

            logger.info("Starting SMILES Optimization")
            optimized_smiles = await self.optimizer.optimize(smiles)
            logger.info(f"SMILES Optimization completed: {optimized_smiles}")

            logger.info("Starting 3D Structure Prediction")
            ligand_pdb = await self.predictor.predict(optimized_smiles, output_dir)
            logger.info(f"3D Structure Prediction completed: {ligand_pdb}")

            logger.info("Starting Ensemble Docking")
            docking_results = await self.ensemble_docker.dock_ensemble(ligand_pdb, protein_ensemble, output_dir)
            logger.info("Ensemble Docking completed")

            logger.info("Starting Docked Ligands Analysis")
            analysis_tasks = [
                self.analyzer.analyze(docking_result['docked_ligand'], protein_ensemble[docking_result['index']], output_dir)
                for docking_result in docking_results if 'docked_ligand' in docking_result
            ]
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Filter out failed analyses
            valid_analysis = []
            for idx, result in enumerate(analysis_results):
                if isinstance(result, dict):
                    valid_analysis.append(result)
                else:
                    logger.warning(f"Analysis task for protein index {idx} failed: {result}")

            logger.info("Docked Ligands Analysis completed")

            logger.info("Starting Ligand Scoring")
            best_ligand = await self.scorer.score(valid_analysis, docking_results)
            logger.info(f"Ligand Scoring completed: {best_ligand}")

            logger.info("Pipeline completed successfully")
            return best_ligand
        except Exception as e:
            logger.error(f"An error occurred in the pipeline: {e}")
            raise

# 9. Usage Example
async def main():
    # Ensure the output directory exists
    output_dir = Path("./SMILESLigand_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # User-provided technical instruction
    technical_instruction = "Design a ligand that binds to the ATP binding site of protein kinase A"

    # List of protein structures (provide actual paths to your protein PDB files)
    protein_ensemble = [
        "./SMILESLigand_results/protein_1.pdb",
        "./SMILESLigand_results/protein_2.pdb",
        # Add more protein structures as needed
    ]

    # Initialize the pipeline
    pipeline = SMILESLigandPipeline()

    # Run the pipeline
    try:
        result = await pipeline.run(
            technical_instruction=technical_instruction,
            protein_ensemble=protein_ensemble,
            output_dir=str(output_dir)
        )
        logger.info(f"Optimized Ligand SMILES: {result.get('smiles')}")
        logger.info(f"Ligand Score: {result.get('score')}")
        # Additional result details can be logged or processed here
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())