import logging
import os
from typing import List, Dict, Any
import sys
from pathlib import Path
from deap import base, creator, tools, algorithms
from rdkit.Chem import Descriptors, AllChem, Crippen
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import mdtraj as md
from openmm import app, unit
from openmm.app import PDBFile, Modeller, Simulation
from openmm import LangevinIntegrator, Platform
import random
from rdkit import Chem
from Bio.PDB import PDBParser
import torch
from colorama import Fore, Style, init
import os
from rdkit import DataStructs
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Initialize colorama
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Checkpoint decorator for synchronous functions
def checkpoint(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Starting step: {step_name}")
            print(Fore.CYAN + f"Starting step: {step_name}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed step: {step_name}")
                print(Fore.GREEN + f"Completed step: {step_name}")
                return result
            except Exception as e:
                logger.error(f"Error in step {step_name}: {str(e)}")
                print(Fore.RED + f"Error in step {step_name}: {str(e)}")
                raise
        return wrapper
    return decorator







VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

def extract_amino_acids(sequence):
    return ''.join(char for char in sequence if char in VALID_AMINO_ACIDS)

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd

# Example SMILES strings
smiles_list = ['CCO', 'CC(=O)O', 'CCN(CC)CC', 'CCCC', 'CC(C)O']

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate descriptors
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol)
    }
    
    # Calculate fingerprints
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    descriptors['Fingerprint'] = list(fp)
    
    return descriptors

# Calculate descriptors for each SMILES string
descriptor_df = pd.DataFrame([calculate_descriptors(s) for s in smiles_list])
print(descriptor_df)








def generate_protein_structures(sequences, output_dir):
    structures = []
    for i, seq in enumerate(sequences):
        try:
            # Create a PDB file for the sequence
            pdb_file = os.path.join(output_dir, f"protein_{i+1}.pdb")
            mol = Chem.MolFromSequence(seq)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            Chem.MolToPDBFile(mol, pdb_file)
            structures.append(pdb_file)
            logger.info(f"Generated structure for sequence {i+1}: {pdb_file}")
        except Exception as e:
            logger.error(f"Error generating structure for sequence {i+1}: {str(e)}")
    return structures



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
        print(Fore.RED + f"Validation error for SMILES {smiles}: {e}")
        return False    

class SMILESGenerator:
    def __init__(self, model_name: str = "laituan245/t5-v1_1-large-caption2smiles"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        if self.tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, model_max_length=1024)
            logger.info("Tokenizer loaded.")
            print(Fore.YELLOW + "Tokenizer loaded.")
        if self.model is None:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on CUDA.")
                print(Fore.YELLOW + "Model loaded on CUDA.")
            else:
                logger.info("Model loaded on CPU.")
                print(Fore.YELLOW + "Model loaded on CPU.")
    
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Model unloaded.")
            print(Fore.YELLOW + "Model unloaded.")
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            logger.info("Tokenizer unloaded.")
            print(Fore.YELLOW + "Tokenizer unloaded.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
            print(Fore.YELLOW + "CUDA cache cleared.")
    

    def refine_smiles(self, initial_smiles: str, max_refinement_attempts: int = 3) -> str:
        for attempt in range(max_refinement_attempts):
            try:
                refinement_prompt = f"Refine and optimize this SMILES string: {initial_smiles}\nOptimized SMILES:"
                input_ids = self.tokenizer(refinement_prompt, return_tensors="pt").input_ids
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                outputs = self.model.generate(input_ids, num_beams=5, max_length=1024)
                refined_smiles = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract SMILES from the refined output
                refined_smiles = refined_smiles.split("Optimized SMILES:")[-1].strip()
                
                if validate_smiles(refined_smiles):
                    logger.info(f"Refined SMILES (attempt {attempt + 1}): {refined_smiles}")
                    return refined_smiles
            except Exception as e:
                logger.warning(f"Refinement attempt {attempt + 1} failed: {str(e)}")
        
        return initial_smiles  # Return the initial SMILES if refinement fails

    def clean_and_validate_smiles(self, smiles: str) -> str:
        # Remove any text before 'SMILES:' if present
        if 'SMILES:' in smiles:
            smiles = smiles.split('SMILES:')[-1].strip()
        
        # Remove any text after the first whitespace
        smiles = smiles.split()[0]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            pass
        
        return ""

    def generate_simple_molecule(self) -> str:
        # Generate a simple, valid molecule as a fallback
        mol = Chem.MolFromSmiles('CC')
        return Chem.MolToSmiles(mol)

    @checkpoint("SMILES Generation")
    def generate(self, technical_instruction: str, max_attempts: int = 5) -> str:
        self.load_model()
        try:
            for attempt in range(max_attempts):
                try:
                    input_ids = self.tokenizer(technical_instruction, return_tensors="pt").input_ids
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                    outputs = self.model.generate(input_ids, num_beams=5, max_length=256)  # Reduced max_length
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    smiles = self.clean_and_validate_smiles(generated_text)
                    if smiles:
                        logger.info(f"Generated valid SMILES: {smiles}")
                        print(Fore.GREEN + f"Generated valid SMILES: {smiles}")
                        return smiles
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            
            # If all attempts fail, generate a simple valid molecule
            simple_smiles = self.generate_simple_molecule()
            logger.warning(f"Failed to generate valid SMILES. Using simple molecule: {simple_smiles}")
            print(Fore.YELLOW + f"Failed to generate valid SMILES. Using simple molecule: {simple_smiles}")
            return simple_smiles
        finally:
            self.unload_model()




    def simplify_smiles(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            return ""

    def optimize_smiles(self, smiles: str, max_attempts: int = 5) -> str:
        for attempt in range(max_attempts):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError("Invalid SMILES")
                
                # Perform simple optimization
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                mol = Chem.RemoveHs(mol)
                
                optimized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                if validate_smiles(optimized_smiles):
                    return optimized_smiles
            except Exception as e:
                logger.warning(f"Optimization attempt {attempt + 1} failed: {str(e)}")
        
        return self.simplify_smiles(smiles)  # Return simplified SMILES if optimization fails




    def generate_smiles_with_protein_context(self, technical_instruction: str, protein_sequences: List[str]) -> str:
        protein_context = " ".join(protein_sequences)
        combined_input = f"{technical_instruction} Protein Sequences: {protein_context} SMILES:"
        return self.generate(combined_input)
    
    
    
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
    def optimize(self, smiles: str) -> str:
        self.original_smiles = smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES provided for optimization.")

        population = self.toolbox.population(n=self.population_size)
        logger.info("Initial population created.")
        print(Fore.CYAN + "Initial population created.")

        # Evaluate the entire population
        for individual in population:
            individual.fitness.values = self.toolbox.evaluate(individual)

        logger.info("Initial population evaluated.")
        print(Fore.CYAN + "Initial population evaluated.")

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
            print(Fore.CYAN + f"Generation {gen + 1} complete.")

            # Early stopping if no improvement
            if all(ind.fitness.values[0] <= 0 for ind in population):
                logger.warning("All individuals have non-positive fitness. Stopping early.")
                print(Fore.YELLOW + "All individuals have non-positive fitness. Stopping early.")
                break

        # Select the best individual
        best_ind = tools.selBest(population, k=1)[0]
        optimized_smiles = best_ind[0]
        logger.info(f"Optimized SMILES: {optimized_smiles}")
        print(Fore.GREEN + f"Optimized SMILES: {optimized_smiles}")
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
                logger.debug(f"Atom removed. New SMILES: {new_smiles}")
                print(Fore.BLUE + f"Atom removed. New SMILES: {new_smiles}")
            except Exception as e:
                logger.warning(f"Mutation failed during atom removal: {e}")
                print(Fore.YELLOW + f"Mutation failed during atom removal: {e}")
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
                logger.debug(f"Atom added. New SMILES: {new_smiles}")
                print(Fore.BLUE + f"Atom added. New SMILES: {new_smiles}")
            except Exception as e:
                logger.warning(f"Mutation failed during atom addition: {e}")
                print(Fore.YELLOW + f"Mutation failed during atom addition: {e}")
        return (individual,)

class StructurePredictor:
    @checkpoint("3D Structure Prediction")
    def predict(self, smiles: str, output_dir: str) -> str:
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
                except Exception as e:
                    logger.warning(f"MMFF optimization failed for conformer {conf_id}: {e}")
                    print(Fore.YELLOW + f"MMFF optimization failed for conformer {conf_id}: {e}")
                    continue

            if not energies:
                raise ValueError("MMFF optimization failed for all conformers.")

            # Select the conformer with the lowest energy
            min_energy_conf = min(energies, key=lambda x: x[0])[1]
            mol.SetProp("_conformerId", str(min_energy_conf))

            ligand_pdb_path = os.path.join(output_dir, f"{smiles[:10]}_ligand.pdb")
            Chem.MolToPDBFile(mol, ligand_pdb_path, confId=min_energy_conf)
            logger.info(f"3D structure saved to: {ligand_pdb_path}")
            print(Fore.GREEN + f"3D structure saved to: {ligand_pdb_path}")
            return ligand_pdb_path
        except Exception as e:
            logger.error(f"Error in 3D structure prediction: {str(e)}")
            print(Fore.RED + f"Error in 3D structure prediction: {str(e)}")
            raise

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
    def dock_ensemble(self, ligand_path: str, protein_ensemble: List[str], output_dir: str) -> List[Dict[str, Any]]:
        docking_results = []
        for i, protein in enumerate(protein_ensemble):
            try:
                result = self.dock_single(ligand_path, protein, output_dir, i)
                docking_results.append(result)
                logger.info(f"Docking completed for protein {i}")
                print(Fore.GREEN + f"Docking completed for protein {i}")
            except Exception as e:
                logger.error(f"Docking failed for protein {i}: {e}")
                print(Fore.RED + f"Docking failed for protein {i}: {e}")
                docking_results.append({"index": i, "affinity": float('inf')})
        return docking_results

    def dock_single(self, ligand_path: str, protein_path: str, output_dir: str, index: int) -> Dict[str, Any]:
        output_path = os.path.join(output_dir, f"docked_{index}.pdbqt")
        log_path = os.path.join(output_dir, f"docking_log_{index}.txt")

        # Prepare receptor and ligand
        try:
            receptor_pdbqt = self._prepare_pdbqt(protein_path, is_receptor=True)
            ligand_pdbqt = self._prepare_pdbqt(ligand_path, is_receptor=False)
        except Exception as e:
            logger.error(f"Preparation failed for docking {index}: {e}")
            print(Fore.RED + f"Preparation failed for docking {index}: {e}")
            return {"index": index, "affinity": float('inf')}

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

        logger.debug(f"Executing command: {cmd}")
        print(Fore.BLUE + f"Executing command: {cmd}")
        os.system(cmd)  # Synchronous execution

        if not os.path.exists(output_path):
            logger.error(f"Docking output not found for protein {index}")
            print(Fore.RED + f"Docking output not found for protein {index}")
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
            print(Fore.RED + f"Failed to parse docking log for protein {index}: {e}")
            best_affinity = float('inf')

        return {"index": index, "affinity": best_affinity, "docked_ligand": output_path}

    def _prepare_pdbqt(self, pdb_path: str, is_receptor: bool) -> str:
        pdbqt_path = pdb_path.replace('.pdb', '.pdbqt')
        if os.path.exists(pdbqt_path):
            return pdbqt_path

        if is_receptor:
            prepare_command = f"prepare_receptor -r {pdb_path} -o {pdbqt_path}"
        else:
            prepare_command = f"prepare_ligand -l {pdb_path} -o {pdbqt_path}"

        logger.debug(f"Executing command: {prepare_command}")
        print(Fore.BLUE + f"Executing command: {prepare_command}")
        os.system(prepare_command)  # Synchronous execution

        if not os.path.exists(pdbqt_path):
            raise RuntimeError(f"Failed to prepare {'receptor' if is_receptor else 'ligand'}.")

        return pdbqt_path

class DockingSimulator:
    @checkpoint("Docking Simulation")
    def dock(self, protein_pdbqt: str, ligand_pdbqt: str, output_dir: str) -> str:
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

            logger.debug(f"Executing command: {docking_command}")
            print(Fore.BLUE + f"Executing command: {docking_command}")
            os.system(docking_command)  # Synchronous execution

            if not os.path.exists(output_path):
                raise RuntimeError("Docking failed.")

            logger.info(f"Docking completed. Output saved to: {output_path}")
            print(Fore.GREEN + f"Docking completed. Output saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error in docking simulation: {str(e)}")
            print(Fore.RED + f"Error in docking simulation: {str(e)}")
            raise

class DockingAnalyzer:
    @checkpoint("Docking Analysis")
    def analyze(self, docked_ligand: str, protein_pdb: str, simulation_output_dir: str) -> Dict[str, Any]:
        try:
            # Run MD simulation
            simulation_output = self._run_md_simulation(docked_ligand, protein_pdb, simulation_output_dir)

            # Analyze trajectory
            results = self._analyze_trajectory(simulation_output, protein_pdb)

            logger.info(f"Analysis completed. Results: {results}")
            print(Fore.GREEN + f"Analysis completed. Results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error in docking analysis: {str(e)}")
            print(Fore.RED + f"Error in docking analysis: {str(e)}")
            raise

    def _run_md_simulation(self, docked_ligand: str, protein_pdb: str, output_dir: str) -> str:
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
        logger.info("Energy minimization completed.")
        print(Fore.GREEN + "Energy minimization completed.")

        # Equilibration
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        simulation.step(10000)  # 20 ps equilibration
        logger.info("Equilibration completed.")
        print(Fore.GREEN + "Equilibration completed.")

        # Production run
        output_file = os.path.join(output_dir, "trajectory.dcd")
        simulation.reporters.append(app.DCDReporter(output_file, 1000))
        simulation.reporters.append(app.StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))
        simulation.step(50000)  # 100 ps production
        logger.info(f"MD simulation completed. Trajectory saved to: {output_file}")
        print(Fore.GREEN + f"MD simulation completed. Trajectory saved to: {output_file}")

        return output_file

    def _analyze_trajectory(self, trajectory_file: str, protein_pdb: str) -> Dict[str, float]:
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

class LigandScorer:
    @checkpoint("Ligand Scoring")
    def score(self, analysis_results: List[Dict[str, Any]], docking_results: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            print(Fore.GREEN + f"Best ligand selected with score: {best_score}")
            return best_ligand
        except Exception as e:
            logger.error(f"Error in ligand scoring: {str(e)}")
            print(Fore.RED + f"Error in ligand scoring: {str(e)}")
            raise

    def _calculate_score(self, analysis_result: Dict[str, Any], docking_result: Dict[str, Any]) -> float:
        # Combine docking affinity and simulation results
        affinity = docking_result.get('affinity', float('inf'))
        avg_rmsd = analysis_result.get('avg_rmsd', float('inf'))
        max_rmsd = analysis_result.get('max_rmsd', float('inf'))

        # Scoring function (weights can be adjusted)
        score = -affinity - avg_rmsd - 0.5 * max_rmsd
        logger.debug(f"Calculated score: {score} (Affinity: {affinity}, Avg RMSD: {avg_rmsd}, Max RMSD: {max_rmsd})")
        print(Fore.BLUE + f"Calculated score: {score} (Affinity: {affinity}, Avg RMSD: {avg_rmsd}, Max RMSD: {max_rmsd})")
        return score
    
    
    def calculate_smiles_score(self, smiles: str) -> float:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                print(Fore.YELLOW + f"Invalid SMILES: {smiles}")
                return float('-inf')

            # Calculate various molecular properties
            mol_weight = Descriptors.ExactMolWt(mol)
            log_p = Crippen.MolLogP(mol)
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            h_bond_donors = Descriptors.NumHDonors(mol)
            h_bond_acceptors = Descriptors.NumHAcceptors(mol)
            polar_surface_area = Descriptors.TPSA(mol)

            # Define ideal ranges for each property
            ideal_mol_weight = (300, 500)
            ideal_log_p = (1, 5)
            ideal_rotatable_bonds = (0, 10)
            ideal_h_bond_donors = (0, 5)
            ideal_h_bond_acceptors = (0, 10)
            ideal_polar_surface_area = (0, 140)

            # Calculate scores for each property
            weight_score = self._range_score(mol_weight, ideal_mol_weight)
            log_p_score = self._range_score(log_p, ideal_log_p)
            rotatable_score = self._range_score(num_rotatable_bonds, ideal_rotatable_bonds)
            donor_score = self._range_score(h_bond_donors, ideal_h_bond_donors)
            acceptor_score = self._range_score(h_bond_acceptors, ideal_h_bond_acceptors)
            psa_score = self._range_score(polar_surface_area, ideal_polar_surface_area)

            # Combine scores (you can adjust weights as needed)
            total_score = (
                weight_score * 0.2 +
                log_p_score * 0.2 +
                rotatable_score * 0.15 +
                donor_score * 0.15 +
                acceptor_score * 0.15 +
                psa_score * 0.15
            )

            logger.info(f"SMILES {smiles} scored: {total_score}")
            print(Fore.GREEN + f"SMILES {smiles} scored: {total_score}")
            return total_score

        except Exception as e:
            logger.error(f"Error calculating SMILES score for {smiles}: {str(e)}")
            print(Fore.RED + f"Error calculating SMILES score for {smiles}: {str(e)}")
            return float('-inf')

    def _range_score(self, value: float, ideal_range: tuple) -> float:
        """
        Calculate a score based on how close a value is to an ideal range.
        Returns 1.0 if the value is within the range, and decreases as it moves away from the range.
        """
        min_val, max_val = ideal_range
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            return max(0, 1 - (min_val - value) / min_val)
        else:
            return max(0, 1 - (value - max_val) / max_val)




import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class MCTSNode:
    def __init__(self, smiles, parent=None):
        self.smiles = smiles
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def is_fully_expanded(self):
        return len(self.children) >= 5

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.reward / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]


class AdditionalLigandOptimizer:
    def __init__(self):
            pass

    @checkpoint("Chemical Property Refinement")
    def refine_properties(self, smiles: str) -> str:
        """
        Refine SMILES based on chemical properties like Lipinski's Rule of Five.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string for property refinement.")

            # Calculate properties
            mw = Descriptors.MolWt(mol)
            log_p = Crippen.MolLogP(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)

            # Apply Lipinski's Rule of Five
            if mw > 500:
                raise ValueError("Molecular weight exceeds Lipinski's rule.")
            if log_p > 5:
                raise ValueError("LogP exceeds Lipinski's rule.")
            if h_donors > 5 or h_acceptors > 10:
                raise ValueError("H-bond donors/acceptors exceed Lipinski's rule.")

            logger.info(f"SMILES {smiles} passed Lipinski's Rule of Five.")
            print(Fore.GREEN + f"SMILES {smiles} passed Lipinski's Rule of Five.")
            return smiles

        except Exception as e:
            logger.warning(f"Property refinement failed for SMILES {smiles}: {e}")
            print(Fore.YELLOW + f"Property refinement failed for SMILES {smiles}: {e}")
            return smiles  # Return original SMILES if refinement fails

    @checkpoint("Stereochemistry Correction")
    def correct_stereochemistry(self, smiles: str) -> str:
        """
        Correct or enhance stereochemistry in the SMILES string.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string for stereochemistry correction.")

            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            smiles_corrected = Chem.MolToSmiles(mol, isomericSmiles=True)
            logger.info(f"Stereochemistry corrected for SMILES: {smiles_corrected}")
            print(Fore.GREEN + f"Stereochemistry corrected for SMILES: {smiles_corrected}")
            return smiles_corrected

        except Exception as e:
            logger.warning(f"Stereochemistry correction failed for SMILES {smiles}: {e}")
            print(Fore.YELLOW + f"Stereochemistry correction failed for SMILES {smiles}: {e}")
            return smiles  # Return original SMILES if correction fails

    @checkpoint("Duplicate Removal")
    def remove_duplicates(self, smiles_list: List[str]) -> List[str]:
        """
        Remove duplicate SMILES from the list.
        """
        unique_smiles = list(set(smiles_list))
        logger.info(f"Removed duplicates. {len(unique_smiles)} unique SMILES remaining.")
        print(Fore.GREEN + f"Removed duplicates. {len(unique_smiles)} unique SMILES remaining.")
        return unique_smiles

    @checkpoint("Similarity Filtering")
    def filter_similarity(self, original_smiles: str, smiles_list: List[str], threshold: float = 0.7) -> List[str]:
        """
        Filter SMILES based on similarity to the original SMILES.
        """
        try:
            original_mol = Chem.MolFromSmiles(original_smiles)
            if original_mol is None:
                raise ValueError("Invalid original SMILES for similarity filtering.")

            original_fp = AllChem.GetMorganFingerprintAsBitVect(original_mol, radius=2)

            filtered_smiles = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
                similarity = DataStructs.TanimotoSimilarity(original_fp, fp)
                if similarity >= threshold:
                    filtered_smiles.append(smiles)

            logger.info(f"Similarity filtering: {len(filtered_smiles)} SMILES passed the threshold of {threshold}.")
            print(Fore.GREEN + f"Similarity filtering: {len(filtered_smiles)} SMILES passed the threshold of {threshold}.")
            return filtered_smiles

        except Exception as e:
            logger.warning(f"Similarity filtering failed: {e}")
            print(Fore.YELLOW + f"Similarity filtering failed: {e}")
            return smiles_list  # Return original list if filtering fails



    def mcts(self, root, iterations=10):
        for _ in range(iterations):
            node = self.tree_policy(root)
            reward = self.default_policy(node)
            self.backpropagate(node, reward)
        return root.best_child(c_param=0)

    def tree_policy(self, node):
        while not node.is_fully_expanded():
            if not node.children:
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        new_smiles = self.mutate_smiles(node.smiles)
        child_node = MCTSNode(new_smiles, parent=node)
        node.children.append(child_node)
        return child_node

    def default_policy(self, node):
        return self.fitness_function(node.smiles)[0]

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def mutate_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        
        # Implement simple mutations (e.g., add/remove atom, change bond)
        mutation_type = np.random.choice(['add', 'remove', 'change'])
        
        if mutation_type == 'add':
            atom = Chem.Atom(6)  # Add a carbon atom
            mol.AddAtom(atom)
        elif mutation_type == 'remove':
            if mol.GetNumAtoms() > 1:
                atom_idx = np.random.randint(mol.GetNumAtoms())
                mol.RemoveAtom(atom_idx)
        elif mutation_type == 'change':
            atom_idx = np.random.randint(mol.GetNumAtoms())
            atom = mol.GetAtomWithIdx(atom_idx)
            new_atomic_num = np.random.choice([6, 7, 8])  # Change to C, N, or O
            atom.SetAtomicNum(new_atomic_num)
        
        return Chem.MolToSmiles(mol)

    def fitness_function(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1, {}
        
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        psa = Descriptors.TPSA(mol)
        
        # Simple scoring based on Lipinski's Rule of Five
        score = (
            (mw <= 500) +
            (logp <= 5) +
            (hbd <= 5) +
            (hba <= 10) +
            (psa <= 140)
        )
        
        properties = {
            'MW': mw,
            'LogP': logp,
            'HBD': hbd,
            'HBA': hba,
            'PSA': psa
        }
        
        return score, properties

    def optimize_smiles(self, smiles, iterations=50, mcts_iterations=10):
        root = MCTSNode(smiles)
        for _ in range(iterations):
            best_node = self.mcts(root, mcts_iterations)
            if best_node.smiles != root.smiles:
                root = MCTSNode(best_node.smiles)
        
        final_score, properties = self.fitness_function(root.smiles)
        logger.info(f"Optimized SMILES: {root.smiles}")
        logger.info(f"Final score: {final_score}")
        logger.info(f"Properties: {properties}")
        return root.smiles

    def refine_properties(self, smiles):
        return self.optimize_smiles(smiles)




# AutoMol-v2/automol/phase2/phase2b/SMILESLigandPipeline.py

import logging
from colorama import Fore, Style, init
from typing import List, Dict, Any
import os

# Initialize colorama
init(autoreset=True)


class SMILESLigandPipeline:
    def __init__(self):
        self.generator = SMILESGenerator()
        self.optimizer = SMILESOptimizer()
        self.predictor = StructurePredictor()
        self.ensemble_docker = EnsembleDocking()
        self.analyzer = DockingAnalyzer()
        self.scorer = LigandScorer()
        self.additional_optimizer = AdditionalLigandOptimizer()  # Initialize the additional optimizer

    def generate_protein_structures(self, protein_sequences, predicted_structures_dir):
        return generate_protein_structures(protein_sequences, predicted_structures_dir)

    @checkpoint("Run SMILES Ligand Pipeline")
    def run_smiles_ligand_pipeline(
        self,
        technical_descriptions: List[str],
        predicted_structures_dir: str,
        results_dir: str,
        num_sequences: int,
        optimization_steps: int,
        score_threshold: float,
        protein_sequences: List[str]
    ) -> List[Dict[str, Any]]:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(predicted_structures_dir, exist_ok=True)
        all_results = []
        best_ligand_overall = None
        best_score_overall = float('-inf')

        
        
        for technical_instruction in technical_descriptions:
            try:
                # SMILES Generation using provided protein_sequences as context
                smiles = self.generator.generate_smiles_with_protein_context(technical_instruction, protein_sequences)

                # 1. Chemical Property Refinement
                refined_smiles = self.additional_optimizer.refine_properties(smiles)
                print(Fore.BLUE + f"Chemical Property Refinement completed: {refined_smiles}")
                logger.info(f"Chemical Property Refinement completed: {refined_smiles}")

                # 2. Stereochemistry Correction
                stereochem_corrected_smiles = self.additional_optimizer.correct_stereochemistry(refined_smiles)
                print(Fore.BLUE + f"Stereochemistry Correction completed: {stereochem_corrected_smiles}")
                logger.info(f"Stereochemistry Correction completed: {stereochem_corrected_smiles}")

                # Update optimized_smiles with refined version
                optimized_smiles = stereochem_corrected_smiles

                # 3. Validate the final optimized SMILES
                if not validate_smiles(optimized_smiles):
                    logger.info("Invalid SMILES string after optimization.")
                    print(Fore.YELLOW + "Invalid SMILES string after optimization.")
                    continue

                # 4. Calculate Score for the Optimized SMILES
                current_score = self.scorer.calculate_smiles_score(optimized_smiles)
                logger.info(f"Calculated score for SMILES: {current_score}")
                print(Fore.GREEN + f"Calculated score for SMILES: {current_score}")

                # 5. 3D Structure Prediction
                print(Fore.CYAN + "Starting 3D Structure Prediction")
                logger.info("Starting 3D Structure Prediction")
                ligand_pdb = self.predictor.predict(optimized_smiles, predicted_structures_dir)
                print(Fore.GREEN + f"3D Structure Prediction completed: {ligand_pdb}")
                logger.info(f"3D Structure Prediction completed: {ligand_pdb}")

                # 6. Generate Protein Structures (Use provided protein_sequences)
                protein_ensemble = self.generate_protein_structures(protein_sequences, predicted_structures_dir)
                print(Fore.GREEN + f"Generated {len(protein_ensemble)} protein structures")
                logger.info(f"Generated {len(protein_ensemble)} protein structures")

                # 7. Ensemble Docking
                print(Fore.CYAN + "Starting Ensemble Docking")
                logger.info("Starting Ensemble Docking")
                docking_results = self.ensemble_docker.dock_ensemble(ligand_pdb, protein_ensemble, results_dir)
                print(Fore.GREEN + "Ensemble Docking completed")
                logger.info("Ensemble Docking completed")

                # 8. Docked Ligands Analysis
                print(Fore.CYAN + "Starting Docked Ligands Analysis")
                logger.info("Starting Docked Ligands Analysis")
                analysis_results = []
                for docking_result in docking_results:
                    if 'docked_ligand' in docking_result:
                        try:
                            analysis_result = self.analyzer.analyze(
                                docking_result['docked_ligand'],
                                protein_ensemble[docking_result['index']],
                                results_dir
                            )
                            analysis_results.append(analysis_result)
                        except Exception as e:
                            logger.warning(f"Analysis failed for docking result: {e}")
                            print(Fore.YELLOW + f"Analysis failed for docking result: {e}")
                print(Fore.GREEN + f"Docked Ligands Analysis completed: {analysis_results}")
                logger.info("Docked Ligands Analysis completed")

                # Filter Valid Analyses
                valid_analysis = [result for result in analysis_results if isinstance(result, dict)]
                for result in analysis_results:
                    if not isinstance(result, dict):
                        logger.warning(f"Analysis task failed: {result}")
                        print(Fore.YELLOW + f"Analysis task failed: {result}")

                # Ligand Scoring
                logger.info("Starting Ligand Scoring")
                print(Fore.CYAN + "Starting Ligand Scoring")
                # Ligand Scoring
                try:
                    best_ligand = self.scorer.score(valid_analysis, docking_results)
                    print(Fore.GREEN + f"Ligand Scoring completed: {best_ligand}")
                    logger.info(f"Ligand Scoring completed: {best_ligand}")

                    if best_ligand["score"] > best_score_overall:
                        best_ligand_overall = best_ligand
                        best_score_overall = best_ligand["score"]

                    if best_ligand["score"] >= score_threshold:
                        all_results.append(best_ligand)
                        logger.info(f"Ligand meets the score threshold: {best_ligand['score']}")
                        print(Fore.GREEN + f"Ligand meets the score threshold: {best_ligand['score']}")
                    else:
                        logger.warning(f"Ligand score below threshold: {best_ligand['score']}")
                        print(Fore.YELLOW + f"Ligand score below threshold: {best_ligand['score']}")
                except Exception as e:
                    logger.warning(f"Ligand scoring failed: {e}")
                    print(Fore.YELLOW + f"Ligand scoring failed: {e}")


            except Exception as e:
                logger.error(f"An error occurred processing instruction '{technical_instruction}': {e}", exc_info=True)
                print(Fore.RED + f"An error occurred processing instruction '{technical_instruction}': {e}")
                logger.info("Continuing to next instruction")
                print(Fore.CYAN + "Continuing to next instruction")

        logger.info("Pipeline completed")
        print(Fore.GREEN + "Pipeline completed")
        return all_results

    def generate_smiles_with_protein_context(self, technical_instruction: str, protein_sequences: List[str]) -> str:
        """
        Generates SMILES strings using protein sequences as context to enhance ligand generation efficacy.
        """
        # Combine technical instruction with protein sequences to provide context
        protein_context = " ".join(protein_sequences)
        combined_input = f"{technical_instruction} Protein Sequences: {protein_context} SMILES:"
        return self.generator.generate(combined_input)