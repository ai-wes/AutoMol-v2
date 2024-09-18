from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rdkit import Chem
import torch

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# Example usage
protein_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
ligand_smiles = generate_ligand_smiles(protein_sequence)

print("Generated valid SMILES:")
for smiles in ligand_smiles:
    print(smiles)



import numpy as np

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

from transformers import T5Tokenizer, T5ForConditionalGeneration

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
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded.")
            print(Fore.YELLOW + "Tokenizer loaded.")
        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
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
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            logger.info("Tokenizer unloaded.")
            print(Fore.YELLOW + "Tokenizer unloaded.")
        if use_cuda:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
            print(Fore.YELLOW + "CUDA cache cleared.")


    def generate(self, protein_sequence, num_sequences=5):
        # Tokenize the input protein sequence
        inputs = self.tokenizer(protein_sequence, return_tensors="pt", truncation=True, padding=True).to(device)

        # Generate SMILES
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_return_sequences=num_sequences,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )

        # Decode and validate SMILES
        generated_smiles = []
        for output in outputs:
            smiles = self.tokenizer.decode(output, skip_special_tokens=True)
            if is_valid_smiles(smiles):
                generated_smiles.append(smiles)

        return generated_smiles


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
    
    
    def mutate_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles  # Return original SMILES if invalid

            mutation_type = random.choice(['add', 'remove', 'change'])

            if mutation_type == 'add':
                atom = Chem.Atom(random.choice([6, 7, 8, 9, 15, 16, 17]))  # C, N, O, F, P, S, Cl
                idx = mol.AddAtom(atom)
                if mol.GetNumAtoms() > 1:
                    bond_idx = random.randint(0, mol.GetNumAtoms() - 2)
                    mol.AddBond(idx, bond_idx, Chem.BondType.SINGLE)
            elif mutation_type == 'remove':
                if mol.GetNumAtoms() > 1:
                    idx_to_remove = random.randint(0, mol.GetNumAtoms() - 1)
                    mol.RemoveAtom(idx_to_remove)
            elif mutation_type == 'change':
                if mol.GetNumAtoms() > 0:
                    idx_to_change = random.randint(0, mol.GetNumAtoms() - 1)
                    atom = mol.GetAtomWithIdx(idx_to_change)
                    new_atomic_num = random.choice([6, 7, 8, 9, 15, 16, 17])  # C, N, O, F, P, S, Cl
                    atom.SetAtomicNum(new_atomic_num)

            try:
                Chem.SanitizeMol(mol)
                new_smiles = Chem.MolToSmiles(mol)
                if validate_smiles(new_smiles):
                    return new_smiles
            except:
                pass
        except Exception as e:
            logger.warning(f"Mutation failed: {e}")

        return smiles  # Return original SMILES if mutation fails

    @checkpoint("MCTS SMILES Optimization")
    def optimize_smiles(self, smiles, iterations=50, mcts_iterations=10):
        root = MCTSNode(smiles)
        for _ in range(iterations):
            best_node = self.mcts(root, mcts_iterations)
            if best_node.smiles != root.smiles and validate_smiles(best_node.smiles):
                root = best_node

        final_score = self.fitness_function(root.smiles)
        logger.info(f"Optimized SMILES: {root.smiles}")
        logger.info(f"Final score: {final_score}")
        return root.smiles



class StructurePredictor:
    @checkpoint("3D Structure Prediction")
    def predict(self, smiles: str, output_dir: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string for structure prediction.")

            mol = Chem.AddHs(mol)
            num_confs = 50
            params = AllChem.ETKDGv3()
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=num_confs,
                params=params,
                pruneRmsThresh=0.5
            )

            if not list(conformer_ids):
                raise ValueError("Embedding failed for the molecule.")

            energies = []
            for conf_id in conformer_ids:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
                    ff.Minimize()
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

            # Save the lowest energy conformer
            mol.SetProp("_conformerId", str(min_energy_conf))
            conf = mol.GetConformer(min_energy_conf)

            ligand_pdb_path = os.path.join(output_dir, f"{smiles[:10]}_ligand.pdb")
            with open(ligand_pdb_path, 'w') as f:
                f.write(Chem.MolToPDBBlock(mol, confId=min_energy_conf))
            logger.info(f"3D structure saved to: {ligand_pdb_path}")
            print(Fore.GREEN + f"3D structure saved to: {ligand_pdb_path}")
            return ligand_pdb_path
        except Exception as e:
            logger.error(f"Error in 3D structure prediction: {str(e)}")
            print(Fore.RED + f"Error in 3D structure prediction: {str(e)}")
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
    def score(self, analysis_results: List[Dict[str, Any]], docking_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Placeholder for scoring implementation
        best_result = max(docking_results, key=lambda x: x['score'])
        logger.info(f"Best docking score: {best_result['score']}")
        return best_result

    def calculate_smiles_score(self, smiles: str) -> float:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return float('-inf')

            mol_weight = Descriptors.ExactMolWt(mol)
            log_p = Descriptors.MolLogP(mol)
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            num_h_donors = Descriptors.NumHDonors(mol)
            num_h_acceptors = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)

            # Scoring based on desired properties
            weight_score = self._range_score(mol_weight, (100, 500))
            log_p_score = self._range_score(log_p, (-1, 5))
            rotatable_score = self._range_score(num_rotatable_bonds, (0, 10))
            donor_score = self._range_score(num_h_donors, (0, 5))
            acceptor_score = self._range_score(num_h_acceptors, (0, 10))
            psa_score = self._range_score(tpsa, (20, 140))

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
    def optimize_smiles(self, smiles, iterations=50, mcts_iterations=10):
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

    def mutate_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        mol = Chem.RWMol(mol)
        mutation_type = np.random.choice(['add', 'remove', 'change'])

        if mutation_type == 'add':
            atom_symbol = np.random.choice(['C', 'N', 'O', 'F', 'Cl', 'Br'])
            new_atom = Chem.Atom(atom_symbol)
            idx = mol.AddAtom(new_atom)
            if mol.GetNumAtoms() > 1:
                bond_idx = random.randint(0, mol.GetNumAtoms() - 2)
                mol.AddBond(idx, bond_idx, Chem.BondType.SINGLE)
        elif mutation_type == 'remove':
            atom_indices = list(range(mol.GetNumAtoms()))
            if len(atom_indices) > 1:
                idx_to_remove = int(np.random.choice(atom_indices))
                mol.RemoveAtom(idx_to_remove)
        elif mutation_type == 'change':
            atom_indices = list(range(mol.GetNumAtoms()))
            idx_to_change = int(np.random.choice(atom_indices))
            atom = mol.GetAtomWithIdx(idx_to_change)
            new_atomic_num = np.random.choice([6, 7, 8, 9, 15, 16, 17])  # Common elements
            atom.SetAtomicNum(int(new_atomic_num))

        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except:
            return smiles

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
                smiles = self.generator.generate(protein_sequences, num_sequences=5)

                # 1. Chemical Property Refinement
                refined_smiles = self.additional_optimizer.refine_properties(smiles)
                print(Fore.BLUE + f"Chemical Property Refinement completed: {refined_smiles}")
                logger.info(f"Chemical Property Refinement completed: {refined_smiles}")

                # 2. Stereochemistry Correction
                stereochem_corrected_smiles = self.additional_optimizer.correct_stereochemistry(refined_smiles)
                print(Fore.BLUE + f"Stereochemistry Correction completed: {stereochem_corrected_smiles}")
                logger.info(f"Stereochemistry Correction completed: {stereochem_corrected_smiles}")

                # 3. MCTS Optimization
                optimized_smiles = self.additional_optimizer.optimize_smiles(stereochem_corrected_smiles)
                print(Fore.BLUE + f"MCTS Optimization completed: {optimized_smiles}")
                logger.info(f"MCTS Optimization completed: {optimized_smiles}")

                # Validate the final optimized SMILES
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
                                docking_result['protein_pdb'],
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