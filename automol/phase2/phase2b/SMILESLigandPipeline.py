from rdkit import Chem
from rdkit.Chem import AllChem
from datetime import datetime
import os
import logging
from colorama import Fore
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from datetime import datetime
import os
from colorama import Fore
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.MolStandardize import rdMolStandardize
import random


import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

import random
import numpy as np

from deap import base, creator, tools, algorithms
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, Crippen
from transformers import EncoderDecoderModel, RobertaTokenizer, pipeline
import gc

from utils.pre_screen_compounds import pre_screen_ligand, screen_and_store_ligand_mongo
import torch
from colorama import Fore, Style, init

# Ensure KMP duplicate lib ok
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Initialize colorama
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if CUDA is available
use_cuda = torch.cuda.is_available()


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


# Function to handle valence errors
def handle_valence_errors(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        return smiles
    except Exception as e:
        logger.warning(f"Valence error for SMILES {smiles}: {e}")
        return None

# Update the mutation functions to handle valence errors
def mutate_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    # Example mutation: Add a random atom
    atom = Chem.Atom(random.choice([6, 7, 8, 9, 15, 16, 17]))  # C, N, O, F, P, S, Cl
    rwmol = Chem.RWMol(mol)
    rwmol.AddAtom(atom)
    new_smiles = Chem.MolToSmiles(rwmol)
    return handle_valence_errors(new_smiles)

# Example usage in the pipeline
def run_smiles_generation_pipeline(initial_smiles: str, generations: int) -> List[str]:
    current_smiles = initial_smiles
    all_smiles = [current_smiles]

    for generation in range(generations):
        new_smiles = mutate_smiles(current_smiles)
        if new_smiles:
            all_smiles.append(new_smiles)
            current_smiles = new_smiles
        logger.info(f"Generation {generation + 1} complete.")

    return all_smiles



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
        n_rotatable = Descriptors.NumRotatableBonds(mol)
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


def generate_protein_structures(sequences, output_dir):
    structures = []
    for i, seq in enumerate(sequences):
        try:
            # Placeholder for actual protein structure generation
            pdb_file = os.path.join(output_dir, f"protein_{i+1}.pdb")
            with open(pdb_file, 'w') as f:
                f.write(f">Protein_{i+1}\n{seq}")
            structures.append(pdb_file)
            logger.info(f"Generated structure for sequence {i+1}: {pdb_file}")
        except Exception as e:
            logger.error(f"Error generating structure for sequence {i+1}: {str(e)}")
    return structures


class SMILESGenerator:
    def __init__(self, model_name: str = "gokceuludogan/WarmMolGenTwo"):
        self.model_name = model_name
        self.protein_tokenizer = None
        self.mol_tokenizer = None
        self.model = None

    def load_model(self):
        if self.protein_tokenizer is None:
            self.protein_tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            logger.info("Protein tokenizer loaded.")
            print(Fore.YELLOW + "Protein tokenizer loaded.")
        
        if self.mol_tokenizer is None:
            self.mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
            logger.info("Molecule tokenizer loaded.")
            print(Fore.YELLOW + "Molecule tokenizer loaded.")
        
        if self.model is None:
            self.model = EncoderDecoderModel.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode
            if use_cuda:
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
        if self.protein_tokenizer is not None:
            del self.protein_tokenizer
            self.protein_tokenizer = None
            logger.info("Protein tokenizer unloaded.")
            print(Fore.YELLOW + "Protein tokenizer unloaded.")
        if self.mol_tokenizer is not None:
            del self.mol_tokenizer
            self.mol_tokenizer = None
            logger.info("Molecule tokenizer unloaded.")
            print(Fore.YELLOW + "Molecule tokenizer unloaded.")
        if use_cuda:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
            print(Fore.YELLOW + "CUDA cache cleared.")

    def is_valid_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    @checkpoint("SMILES Generation")
    def generate(self, protein_sequence: str, num_sequences: int = 5) -> List[str]:
        self.load_model()
        try:
            inputs = self.protein_tokenizer(protein_sequence, return_tensors="pt")
            if use_cuda:
                inputs = inputs.to('cuda')
            
            outputs = self.model.generate(
                **inputs,
                decoder_start_token_id=self.mol_tokenizer.bos_token_id,
                eos_token_id=self.mol_tokenizer.eos_token_id,
                pad_token_id=self.mol_tokenizer.eos_token_id,
                max_length=256,
                num_return_sequences=num_sequences,
                do_sample=True,
                top_p=0.95
            )

            generated_smiles = self.mol_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            logger.info("Generated SMILES:")
            valid_smiles = []
            for smiles in generated_smiles:
                is_valid = self.is_valid_smiles(smiles)
                logger.info(f"{smiles}: {is_valid}")
                print(f"{smiles}: {is_valid}")
                if is_valid:
                    valid_smiles.append(smiles)

            return valid_smiles

        except Exception as e:
            logger.error(f"Error in SMILES generation: {str(e)}")
            print(Fore.RED + f"Error in SMILES generation: {str(e)}")
            return []

        finally:
            print(Fore.YELLOW + "Unloading model...")
            self.unload_model()
            torch.cuda.empty_cache()
            gc.collect()
            print(Fore.YELLOW + "CUDA cache cleared.")

class SMILESOptimizer:
    def __init__(self, population_size=50, generations=20):
        self.population_size = population_size
        self.generations = generations

        # Define a single-objective fitness function to maximize LogP
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
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


    def validate_smiles(self, smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Check for valid atom types
            valid_atoms = set(['C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'])
            if not all(atom.GetSymbol() in valid_atoms for atom in mol.GetAtoms()):
                logger.debug(f"SMILES contains invalid atom types: {smiles}")
                return False

            # Check molecular weight
            mol_weight = Descriptors.ExactMolWt(mol)
            if mol_weight < 100 or mol_weight > 1000:
                logger.debug(f"Molecular weight out of range ({mol_weight}): {smiles}")
                return False

            # Check number of rotatable bonds
            n_rotatable = Descriptors.NumRotatableBonds(mol)
            if n_rotatable > 10:
                logger.debug(f"Too many rotatable bonds ({n_rotatable}): {smiles}")
                return False

            # Check for kekulization
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except:
                logger.debug(f"Kekulization failed for SMILES: {smiles}")
                return False

            return True
        except Exception as e:
            logger.error(f"Validation error for SMILES {smiles}: {e}")
            print(Fore.RED + f"Validation error for SMILES {smiles}: {e}")
            return False



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
                Chem.SanitizeMol(mol)
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
            idx = mol.AddAtom(new_atom)
            # Connect the new atom to a random existing atom
            if mol.GetNumAtoms() > 1:
                bond_idx = random.randint(0, mol.GetNumAtoms() - 2)
                mol.AddBond(idx, bond_idx, Chem.BondType.SINGLE)
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
    
    

    def mutate_smiles(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES string: {smiles}")
                return smiles  # Return original SMILES if invalid

            mutation_type = random.choice(['add', 'remove', 'change'])

            if mutation_type == 'add':
                atom = Chem.Atom(random.choice([6, 7, 8, 9, 15, 16, 17]))  # C, N, O, F, P, S, Cl
                rwmol = Chem.RWMol(mol)
                idx = rwmol.AddAtom(atom)
                if rwmol.GetNumAtoms() > 1:
                    bond_idx = random.randint(0, rwmol.GetNumAtoms() - 2)
                    rwmol.AddBond(idx, bond_idx, Chem.BondType.SINGLE)
                new_mol = rwmol.GetMol()
            elif mutation_type == 'remove':
                if mol.GetNumAtoms() > 1:
                    idx_to_remove = random.randint(0, mol.GetNumAtoms() - 1)
                    rwmol = Chem.RWMol(mol)
                    rwmol.RemoveAtom(idx_to_remove)
                    new_mol = rwmol.GetMol()
                else:
                    return smiles
            elif mutation_type == 'change':
                if mol.GetNumAtoms() > 0:
                    idx_to_change = random.randint(0, mol.GetNumAtoms() - 1)
                    atom = mol.GetAtomWithIdx(idx_to_change)
                    new_atomic_num = random.choice([6, 7, 8, 9, 15, 16, 17])  # C, N, O, F, P, S, Cl
                    atom.SetAtomicNum(new_atomic_num)
                new_mol = mol
            else:
                return smiles

            try:
                Chem.SanitizeMol(new_mol)
                new_smiles = Chem.MolToSmiles(new_mol)
                if self.validate_smiles(new_smiles):
                    logger.debug(f"Mutation successful. New SMILES: {new_smiles}")
                    print(Fore.BLUE + f"Mutation successful. New SMILES: {new_smiles}")
                    return new_smiles
                else:
                    logger.debug(f"Mutation resulted in invalid SMILES: {new_smiles}")
                    print(Fore.YELLOW + f"Mutation resulted in invalid SMILES: {new_smiles}")
                    return smiles
            except Exception as e:
                logger.warning(f"Mutation failed: {e}")
                print(Fore.YELLOW + f"Mutation failed: {e}")
                return smiles

        except Exception as e:
            logger.warning(f"Mutation process encountered an error: {e}")
            print(Fore.YELLOW + f"Mutation process encountered an error: {e}")
            return smiles  # Return original SMILES if mutation fails


    def mutate_molecule(self, individual):
        smiles = individual[0]
        new_smiles = self.mutate_smiles(smiles)
        individual[0] = new_smiles
        return (individual,)

    @checkpoint("MCTS SMILES Optimization")
    def optimize_smiles(self, smiles, iterations=50, mcts_iterations=10):
        root = MCTSNode(smiles)
        for _ in range(iterations):
            best_node = self.mcts(root, mcts_iterations)
            if best_node.smiles != root.smiles and self.validate_smiles(best_node.smiles):
                root = best_node

        final_score = self.fitness_function(root.smiles)
        logger.info(f"Optimized SMILES: {root.smiles}")
        print(Fore.GREEN + f"Optimized SMILES: {root.smiles}")
        return root.smiles


class StructurePredictor:
    @checkpoint("3D Structure Prediction")
    def predict(self, smiles: str, output_dir: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")

            # Add hydrogens to the molecule
            mol = Chem.AddHs(mol)

            # Try to generate 3D conformers
            try:
                confIds = AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
            except Exception as e:
                logger.warning(f"Failed to generate 3D conformers using EmbedMultipleConfs: {e}")
                # Fallback to a simpler 2D to 3D conversion
                AllChem.Compute2DCoords(mol)
                confIds = [0]  # Use the single 2D conformation

            if not confIds:
                raise ValueError("Failed to generate any conformers")

            # Optimize the geometry
            for confId in confIds:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, confId=confId, maxIters=500)
                except Exception as e:
                    logger.warning(f"Failed to optimize conformer {confId}: {e}")

            # Select the lowest energy conformer
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol)
                energies = [ff.CalcEnergy() for _ in range(mol.GetNumConformers())]
                min_energy_conf = min(range(len(energies)), key=energies.__getitem__)
            except Exception as e:
                logger.warning(f"Failed to calculate energies: {e}")
                min_energy_conf = 0  # Use the first conformer if energy calculation fails

            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ligand_{timestamp}.pdb"
            filepath = os.path.join(output_dir, filename)

            # Write the lowest energy conformer to a PDB file
            Chem.MolToPDBFile(mol, filepath, confId=min_energy_conf)

            logger.info(f"3D structure prediction completed. PDB file saved: {filepath}")
            print(Fore.GREEN + f"3D structure prediction completed. PDB file saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error in 3D structure prediction: {str(e)}")
            print(Fore.RED + f"Error in 3D structure prediction: {str(e)}")
            # Return a placeholder or default PDB file path
            return os.path.join(output_dir, "default_ligand.pdb")

    def predict_3d_structure(self, sequence):
        return self.predict(sequence, ".")  # Use current directory as output_dir

    @checkpoint("Protein Structure Prediction")
    def predict_protein_structure(self, sequence: str, output_dir: str) -> str:
        """
        Predict the protein structure from the given sequence and save the PDB file.
        Returns the path to the predicted PDB file.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Structure prediction logic here
            pdb_content = self.predict_3d_structure(sequence)
            pdb_filename = f"{sequence[:10]}_structure.pdb"
            pdb_file_path = os.path.join(output_dir, pdb_filename)
            with open(pdb_file_path, 'w') as pdb_file:
                pdb_file.write(pdb_content)
            return pdb_file_path
        except Exception as e:
            logger.error(f"Failed to predict structure for sequence {sequence}: {e}")
            raise

        
class EnsembleDocking:
    def __init__(self, vina_path: str = "vina", exhaustiveness: int = 8, num_modes: int = 9):
        self.vina_path = vina_path
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes

    def dock_ensemble(self, ligand_pdb: str, protein_ensemble: List[str], results_dir: str) -> List[Dict[str, Any]]:
        # Placeholder for docking implementation
        docking_results = []
        for i, protein_pdb in enumerate(protein_ensemble):
            result = {
                'index': i,
                'protein_pdb': protein_pdb,
                'docked_ligand': ligand_pdb,
                'score': random.uniform(-10, -5)  # Placeholder score
            }
            docking_results.append(result)
            logger.info(f"Docked ligand to protein {i+1}: score {result['score']}")
        return docking_results


class DockingAnalyzer:
    def analyze(self, docked_ligand: str, protein_pdb: str, results_dir: str) -> Dict[str, Any]:
        # Placeholder for analysis implementation
        analysis_result = {
            'docked_ligand': docked_ligand,
            'protein_pdb': protein_pdb,
            'interaction_energy': random.uniform(-50, -10)  # Placeholder interaction energy
        }
        logger.info(f"Analyzed docking: interaction energy {analysis_result['interaction_energy']}")
        return analysis_result


class LigandScorer:
    def calculate_smiles_score(self, smiles: str) -> float:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            mw = Descriptors.ExactMolWt(mol)
            log_p = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            # More nuanced scoring based on Lipinski's Rule of Five and additional criteria
            mw_score = 1.0 - abs(mw - 350) / 150  # Optimal around 350, penalize deviation
            log_p_score = 1.0 - abs(log_p - 2.5) / 3.1  # Optimal around 2.5, penalize deviation
            hbd_score = 1.0 - hbd / 5 if hbd <= 5 else 0.0
            hba_score = 1.0 - hba / 10 if hba <= 10 else 0.0
            tpsa_score = 1.0 - tpsa / 140 if tpsa <= 140 else 0.0
            rotatable_score = 1.0 - rotatable_bonds / 10 if rotatable_bonds <= 10 else 0.0

            # Ensure all scores are between 0 and 1
            scores = [max(0, min(1, score)) for score in [mw_score, log_p_score, hbd_score, hba_score, tpsa_score, rotatable_score]]

            # Calculate the final score as a weighted average of all criteria
            weights = [1.5, 1.5, 1, 1, 1, 1]  # Give slightly more weight to MW and LogP
            score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

            return score

        except Exception as e:
            logger.error(f"Error calculating SMILES score: {str(e)}")
            return 0.0


class MCTSNode:
    def __init__(self, smiles, parent=None):
        self.smiles = smiles
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def is_fully_expanded(self):
        # Assuming each node can have up to 5 children
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
            if mw > 500 or log_p > 5 or h_donors > 5 or h_acceptors > 10:
                # Modify the molecule to try to meet the rules
                # Placeholder: return original SMILES
                logger.warning(f"SMILES {smiles} does not meet Lipinski's Rule of Five.")
                print(Fore.YELLOW + f"SMILES {smiles} does not meet Lipinski's Rule of Five.")
                return smiles

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




    @checkpoint("MCTS SMILES Optimization")
    def optimize_smiles_mcts(self, smiles, iterations=50, mcts_iterations=10):
        root = MCTSNode(smiles)
        for _ in range(iterations):
            best_node = self.mcts(root, mcts_iterations)
            if best_node.smiles != root.smiles:
                root = best_node

        final_score = self.fitness_function(root.smiles)
        logger.info(f"Optimized SMILES: {root.smiles}")
        print(Fore.GREEN + f"Optimized SMILES: {root.smiles}")
        return root.smiles

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
        return self.fitness_function(node.smiles)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def mcts(self, root, iterations=10):
        for _ in range(iterations):
            node = self.tree_policy(root)
            reward = self.default_policy(node)
            self.backpropagate(node, reward)
        return root.best_child(c_param=0)


    def fitness_function(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1.0

        mw = Descriptors.ExactMolWt(mol)
        log_p = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)

        # Simple scoring based on Lipinski's Rule of Five
        score = (
            (mw <= 500) +
            (log_p <= 5) +
            (hbd <= 5) +
            (hba <= 10) +
            (tpsa <= 140)
        )
        return score



class SMILESLigandPipeline:
    def __init__(self):
        self.generator = SMILESGenerator()
        self.optimizer = SMILESOptimizer()
        self.predictor = StructurePredictor()
        self.ensemble_docker = EnsembleDocking()
        self.analyzer = DockingAnalyzer()
        self.scorer = LigandScorer()
        self.additional_optimizer = AdditionalLigandOptimizer()

    
    
    def validate_novel_smiles(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return True
        except:
            return False

    def generate_novel_smiles(self, protein_sequence: str, num_sequences: int) -> List[str]:
        valid_smiles = []
        attempts = 0
        max_attempts = num_sequences * 10  # Allow more attempts for novel molecules
        while len(valid_smiles) < num_sequences and attempts < max_attempts:
            smiles = self.generator.generate(protein_sequence, 1)[0]
            if self.validate_novel_smiles(smiles):
                valid_smiles.append(smiles)
            attempts += 1
            if attempts % 10 == 0:
                logger.info(f"Attempted {attempts} times to generate valid SMILES. Found {len(valid_smiles)} valid SMILES so far.")
                print(Fore.YELLOW + f"Attempted {attempts} times to generate valid SMILES. Found {len(valid_smiles)} valid SMILES so far.")
        return valid_smiles

    def score_novel_ligand(self, ligand, protein):
        # Basic physicochemical properties
        mw = Descriptors.ExactMolWt(ligand)
        logp = Crippen.MolLogP(ligand)
        hbd = Descriptors.NumHDonors(ligand)
        hba = Descriptors.NumHAcceptors(ligand)
        
        # Structural complexity
        complexity = Descriptors.BertzCT(ligand)
        
        # Docking score (if available)
        docking_score = self.scorer.calculate_docking_score(ligand, protein)
        
        # Combine scores (adjust weights as needed)
        score = (
            -0.1 * abs(mw - 400) +  # Prefer MW around 400
            -0.5 * abs(logp - 2.5) +  # Prefer LogP around 2.5
            -0.2 * max(hbd - 5, 0) +  # Penalize excessive H-bond donors
            -0.2 * max(hba - 10, 0) +  # Penalize excessive H-bond acceptors
            0.01 * complexity +  # Slight preference for complexity
            -1.0 * docking_score  # Lower docking score is better
        )
        return score



    def adjust_logp(self, smiles, target_logp=2.5, tolerance=0.5):
        
        
        """
        Adjust the LogP of the compound by adding hydrophilic or lipophilic groups.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        current_logp = Crippen.MolLogP(mol)
        
        while abs(current_logp - target_logp) > tolerance:
            if current_logp < target_logp:
                # Add a lipophilic group (e.g., methyl)
                mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles('[H]'), 
                                                Chem.MolFromSmiles('C'), 
                                                replacementType='random')[0]
            else:
                # Add a hydrophilic group (e.g., hydroxyl)
                mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles('[H]'), 
                                                Chem.MolFromSmiles('O'), 
                                                replacementType='random')[0]

            current_logp = Crippen.MolLogP(mol)

        return Chem.MolToSmiles(mol)



    def optimize_smiles(self, smiles):
        """
        Apply a series of optimization steps to the SMILES.
        """
        smiles = self.standardize_molecule(smiles)
        smiles = self.fragment_based_optimization(smiles)
        smiles = self.bioisostere_replacement(smiles)
        smiles = self.adjust_molecular_weight(smiles)
        smiles = self.adjust_logp(smiles)
        return smiles


    @checkpoint("Run SMILES Ligand Pipeline")
    def run_smiles_ligand_pipeline(
        self,
        predicted_structures_dir: str,
        results_dir: str,
        num_sequences: int,
        optimization_steps: int,
        score_threshold: float,
        protein_sequences: List[str]
    ) -> List[Dict[str, Any]]:
        all_results = []

        for protein_sequence in protein_sequences:
            try:
                smiles_list = self.generator.generate(protein_sequence, num_sequences)
                logger.info(f"Generated SMILES: {smiles_list}")
                print(Fore.CYAN + f"Generated SMILES: {smiles_list}")

                for smiles in smiles_list:
                    result = self.process_single_smiles(smiles, protein_sequence, predicted_structures_dir, results_dir, score_threshold)
                    if result:
                        all_results.append(result)

            except Exception as e:
                logger.error(f"Error processing protein sequence {protein_sequence}: {e}", exc_info=True)
                print(Fore.RED + f"Error processing protein sequence {protein_sequence}: {e}")

        logger.info("Pipeline completed")
        print(Fore.CYAN + "Pipeline completed")

        return all_results
    
    

    def process_single_smiles(self, smiles: str, protein_sequence: str, predicted_structures_dir: str, results_dir: str, score_threshold: float) -> Dict[str, Any]:
        result = {"smiles": smiles}
        
        # 1. Optimize SMILES
        try:
            optimized_smiles = self.optimizer.optimize_smiles(smiles)
            result['optimized_smiles'] = optimized_smiles
            logger.info(f"Optimized SMILES: {optimized_smiles}")
            print(Fore.GREEN + f"Optimized SMILES: {optimized_smiles}")
        except Exception as e:
            logger.warning(f"Error during SMILES optimization: {e}")
            print(Fore.YELLOW + f"Error during SMILES optimization: {e}")
            return None
        
        # 2. Validate and Adjust LogP
        try:
            adjusted_smiles = self.adjust_logp(optimized_smiles)
            result['adjusted_smiles'] = adjusted_smiles
            logger.info(f"Adjusted SMILES: {adjusted_smiles}")
            print(Fore.GREEN + f"Adjusted SMILES: {adjusted_smiles}")
        except Exception as e:
            logger.warning(f"Error during LogP adjustment: {e}")
            print(Fore.YELLOW + f"Error during LogP adjustment: {e}")
            return None
        
        # 3. 3D Structure Prediction
        try:
            ligand_pdb = self.predictor.predict(adjusted_smiles, predicted_structures_dir)
            result['ligand_pdb'] = ligand_pdb
            logger.info(f"3D Structure Prediction completed: {ligand_pdb}")
            print(Fore.GREEN + f"3D Structure Prediction completed: {ligand_pdb}")
        except Exception as e:
            logger.warning(f"Error in 3D structure prediction: {e}")
            print(Fore.YELLOW + f"Error in 3D structure prediction: {e}")
            return None

        # 4. Generate Protein Structures
        try:
            protein_ensemble = self.generate_protein_structures([protein_sequence], predicted_structures_dir)
            logger.info(f"Generated {len(protein_ensemble)} protein structures")
            print(Fore.GREEN + f"Generated {len(protein_ensemble)} protein structures")
        except Exception as e:
            logger.warning(f"Error in protein structure generation: {e}")
            print(Fore.YELLOW + f"Error in protein structure generation: {e}")
            return None

        # 5. Ensemble Docking
        if result.get('ligand_pdb') and protein_ensemble:
            try:
                docking_results = self.ensemble_docker.dock_ensemble(result['ligand_pdb'], protein_ensemble, results_dir)
                result['docking_results'] = docking_results
                logger.info("Ensemble Docking completed")
                print(Fore.GREEN + "Ensemble Docking completed")
            except Exception as e:
                logger.warning(f"Error in ensemble docking: {e}")
                print(Fore.YELLOW + f"Error in ensemble docking: {e}")
                return None

        # 6. Analyze Docking Results
        if result.get('docking_results'):
            try:
                analysis = self.analyzer.analyze(docking_results)
                result['analysis'] = analysis
                logger.info("Docking Analysis completed")
                print(Fore.GREEN + "Docking Analysis completed")
            except Exception as e:
                logger.warning(f"Error in docking analysis: {e}")
                print(Fore.YELLOW + f"Error in docking analysis: {e}")
                return None

        # 7. Scoring
        try:
            score = self.score_novel_ligand(result['docking_results'], protein_ensemble)
            result['score'] = score
            logger.info(f"Ligand scored: {score}")
            print(Fore.GREEN + f"Ligand scored: {score}")
        except Exception as e:
            logger.warning(f"Error in scoring: {e}")
            print(Fore.YELLOW + f"Error in scoring: {e}")
            return None

        # 8. Filter based on score
        if score >= score_threshold:
            logger.info(f"Ligand {smiles} passed with score {score}")
            print(Fore.GREEN + f"Ligand {smiles} passed with score {score}")
            return result
        else:
            logger.info(f"Ligand {smiles} did not meet the score threshold ({score_threshold}).")
            print(Fore.YELLOW + f"Ligand {smiles} did not meet the score threshold ({score_threshold}).")
            return None
        
        
        
        
    @checkpoint("Iterative Optimization")
    def iterative_optimization(self, smiles, max_iterations=10):
        """
        Iteratively apply optimizations until pre-screening criteria are met or max iterations reached.
        """
        for i in range(max_iterations):
            try:
                optimized_smiles = self.optimizer.optimize(smiles)
                iterative_optimized_smiles = self.additional_optimizer.optimize_smiles(optimized_smiles)
                passed, message = pre_screen_ligand(iterative_optimized_smiles)
                if passed:
                    logging.info(f"SMILES passed pre-screening after {i+1} iterations: {iterative_optimized_smiles}")
                    print(Fore.GREEN + f"SMILES passed pre-screening after {i+1} iterations: {iterative_optimized_smiles}")
                    return iterative_optimized_smiles
                smiles = iterative_optimized_smiles
            except Exception as e:
                logging.warning(f"Error during optimization iteration {i+1}: {e}")
                print(Fore.YELLOW + f"Error during optimization iteration {i+1}: {e}")
                # If an error occurs, continue with the original SMILES
                continue

        logging.warning(f"SMILES failed to pass pre-screening after {max_iterations} iterations: {smiles}")
        print(Fore.YELLOW + f"SMILES failed to pass pre-screening after {max_iterations} iterations: {smiles}")
        return smiles  # Return the last optimized version even if it didn't pass
    
    
    
    
    def standardize_molecule(self, smiles):
        """
        Standardize the molecule using RDKit's MolStandardize if available.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES for standardization: {smiles}")
            return smiles

        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize
            standardizer = rdMolStandardize.Standardizer()
            mol = standardizer.standardize(mol)
            standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return standardized_smiles
        except AttributeError:
            logger.warning("RDKit's Standardizer not available. Using a simple standardization approach.")
            # Simple standardization: remove hydrogens and regenerate SMILES
            mol = Chem.RemoveHs(mol)
            mol = Chem.AddHs(mol)
            standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return standardized_smiles



    @checkpoint("Fragment-Based Optimization")
    def fragment_based_optimization(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        drug_like_fragments = ['c1ccccc1', 'C(=O)N', 'C(=O)O', 'CN', 'CF', 'CCl']

        num_atoms = mol.GetNumAtoms()
        atoms_to_replace = random.sample(range(num_atoms), k=min(3, num_atoms))

        for atom_idx in atoms_to_replace:
            fragment = Chem.MolFromSmiles(random.choice(drug_like_fragments))
            if fragment is not None:
                atom = mol.GetAtomWithIdx(atom_idx)
                rwmol = Chem.RWMol(mol)
                rwmol.ReplaceAtom(atom_idx, fragment.GetAtomWithIdx(0))
                mol = rwmol.GetMol()

        return Chem.MolToSmiles(mol)



    @checkpoint("Bioisostere Replacement")
    def bioisostere_replacement(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        bioisosteres = {
            'C(=O)OH': 'C(=O)NH2',
            'c1ccccc1': 'c1ccncc1',
            'CF': 'CCl',
            'S': 'O',
        }

        for original, replacement in bioisosteres.items():
            pattern = Chem.MolFromSmiles(original)
            replace = Chem.MolFromSmiles(replacement)
            if pattern is not None and replace is not None:
                mol = AllChem.ReplaceSubstructs(mol, pattern, replace, replaceAll=True)[0]

        return Chem.MolToSmiles(mol)