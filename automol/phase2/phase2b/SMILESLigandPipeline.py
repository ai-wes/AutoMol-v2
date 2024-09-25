import os
import sys
import gc
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing import Tuple

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from colorama import Fore, Style, init
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.MolStandardize import rdMolStandardize

from deap import base, creator, tools, algorithms
from transformers import EncoderDecoderModel, RobertaTokenizer

# Ensure KMP duplicate lib is handled
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Initialize colorama
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if CUDA is available
USE_CUDA = torch.cuda.is_available()


def checkpoint(step_name):
    """Decorator for logging the start and completion of steps."""
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


def handle_valence_errors(smiles: str) -> Optional[str]:
    """Handle valence errors during SMILES parsing."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        return smiles
    except Exception as e:
        logger.warning(f"Valence error for SMILES {smiles}: {e}")
        return None


def pre_screen_ligand(smiles: str) -> Tuple[bool, str]:
    """
    Pre-screen the ligand based on basic criteria.
    For demonstration, this function checks if the molecule has less than 50 atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES for minimal viability"

    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    # Less stringent criteria
    weight_range = (100, 600)
    logp_range = (-3, 6)

    if not (weight_range[0] <= mol_weight <= weight_range[1]):
        return False, f"MW {mol_weight} out of minimal range"
    if not (logp_range[0] <= logp <= logp_range[1]):
        return False, f"LogP {logp} out of minimal range"

    return True, "Ligand passes minimal viability screening"



class SMILESGenerator:
    """Generates SMILES strings using a pretrained model."""

    def __init__(self, model_name: str = "gokceuludogan/WarmMolGenTwo"):
        self.model_name = model_name
        self.protein_tokenizer = None
        self.mol_tokenizer = None
        self.model = None

    def load_model(self):
        """Load the tokenizer and model."""
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
            if USE_CUDA:
                self.model = self.model.cuda()
                logger.info("Model loaded on CUDA.")
                print(Fore.YELLOW + "Model loaded on CUDA.")
            else:
                logger.info("Model loaded on CPU.")
                print(Fore.YELLOW + "Model loaded on CPU.")

    def unload_model(self):
        """Unload the model and tokenizer to free up memory."""
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
        if USE_CUDA:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
            print(Fore.YELLOW + "CUDA cache cleared.")

    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if a SMILES string is valid."""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    @checkpoint("SMILES Generation")
    def generate_smiles_from_protein(self, protein_sequence: str, num_sequences: int = 5) -> List[str]:
        """Generate SMILES strings based on a protein sequence."""
        self.load_model()
        try:
            inputs = self.protein_tokenizer(protein_sequence, return_tensors="pt")
            if USE_CUDA:
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
    """Optimizes SMILES strings using genetic algorithms."""

    def __init__(self, population_size=200, generations=10):
        self.population_size = population_size
        self.generations = generations

        # Define a single-objective fitness function to maximize LogP + QED
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
        """Initialize an individual with a random SMILES string."""
        # Placeholder: Initialize with a random SMILES or a specific SMILES
        # Here, we initialize with a generic benzene ring
        return creator.Individual(['c1ccccc1'])

    def optimize_smiles(self, smiles: str) -> str:
        """Optimize a single SMILES string."""
        self.original_smiles = smiles
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
        """Fitness function based on LogP and QED."""
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
        """Crossover operation: swap SMILES strings."""
        ind1[0], ind2[0] = ind2[0], ind1[0]
        return ind1, ind2

    def mutate_smiles(self, smiles: str) -> str:
        """Mutate a SMILES string by adding, removing, or changing atoms."""
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
                if self.is_valid_smiles(new_smiles):
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
        """Apply mutation to an individual."""
        smiles = individual[0]
        new_smiles = self.mutate_smiles(smiles)
        individual[0] = new_smiles
        return (individual,)

    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if the mutated SMILES is valid."""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None


class StructurePredictor:
    """Predicts the 3D structure of ligands and proteins."""

    def __init__(self):
        pass

    @checkpoint("3D Structure Prediction")
    def predict_3d_ligand_structure(self, smiles: str, output_dir: str) -> str:
        """Predict the 3D structure of a ligand and save it as a PDB file."""
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

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

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

    def generate_ligand_structures(self, smiles: str, output_dir: str) -> List[str]:
        """Generate 3D structures for the ligand."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string")

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"ligand_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdb")
            Chem.MolToPDBFile(mol, output_path)
            return [output_path]
        except Exception as e:
            logger.error(f"Error in 3D structure prediction: {e}")
            return ["default_ligand.pdb"]

    @checkpoint("3D Protein Structure Prediction")
    def predict_3d_protein_structure(self, protein_sequence: str, output_dir: str) -> List[str]:
        """
        Predict the 3D structure of a protein from its sequence.
        For demonstration purposes, this function will generate a dummy PDB file.
        In practice, you would integrate with a protein structure prediction tool like AlphaFold.
        """
        try:
            # Placeholder: Generate a dummy PDB file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"protein_{timestamp}.pdb"
            filepath = os.path.join(output_dir, filename)

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            with open(filepath, 'w') as f:
                f.write("ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N\n")
                f.write("ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 20.00           C\n")
                # Add more dummy atoms as needed

            logger.info(f"3D protein structure prediction completed. PDB file saved: {filepath}")
            print(Fore.GREEN + f"3D protein structure prediction completed. PDB file saved: {filepath}")
            return [filepath]

        except Exception as e:
            logger.error(f"Error in 3D protein structure prediction: {e}")
            print(Fore.RED + f"Error in 3D protein structure prediction: {e}")
            return [os.path.join(output_dir, "default_protein.pdb")]


class LigandScorer:
    """Calculates scores for ligands based on various properties."""

    def calculate_smiles_score(self, smiles: str) -> float:
        """Calculate a score for a given SMILES string based on physicochemical properties."""
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
            scores = [max(0, min(1, score)) for score in [
                mw_score, log_p_score, hbd_score, hba_score, tpsa_score, rotatable_score
            ]]

            # Calculate the final score as a weighted average of all criteria
            weights = [1.5, 1.5, 1, 1, 1, 1]  # Give slightly more weight to MW and LogP
            score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

            return score

        except Exception as e:
            logger.error(f"Error calculating SMILES score: {str(e)}")
            return 0.0

    def calculate_docking_score(self, docking_results: List[Dict[str, Any]], protein_ensemble: List[str]) -> float:
        """Calculate an overall docking score based on docking results."""
        if not docking_results:
            return 0.0
        # Example: average docking score
        scores = [result['score'] for result in docking_results if 'score' in result]
        if not scores:
            return 0.0
        average_docking_score = sum(scores) / len(scores)
        return average_docking_score


class EnsembleDocker:
    """Performs ensemble docking of ligands to a protein ensemble."""

    @checkpoint("Ensemble Docking")
    def dock_ensemble(self, ligand_pdb: str, protein_ensemble: List[str], results_dir: str) -> List[Dict[str, Any]]:
        """
        Perform docking of the ligand to each protein structure in the ensemble.
        For demonstration, this function will generate dummy docking results.
        In practice, integrate with a docking tool like AutoDock Vina.
        """
        try:
            docking_results = []
            for protein_pdb in protein_ensemble:
                # Placeholder: Generate a random docking score
                score = random.uniform(-15.0, -5.0)  # Typical binding affinities in kcal/mol
                docking_results.append({'protein': protein_pdb, 'score': score})

            # Save docking results to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filepath = os.path.join(results_dir, f"docking_results_{timestamp}.txt")
            os.makedirs(results_dir, exist_ok=True)
            with open(results_filepath, 'w') as f:
                for result in docking_results:
                    f.write(f"Protein: {result['protein']}, Score: {result['score']}\n")

            logger.info(f"Ensemble docking completed. Results saved to {results_filepath}")
            print(Fore.GREEN + f"Ensemble docking completed. Results saved to {results_filepath}")
            return docking_results

        except Exception as e:
            logger.error(f"Error in ensemble docking: {e}")
            print(Fore.RED + f"Error in ensemble docking: {e}")
            return []


class Analyzer:
    """Analyzes docking results."""

    @checkpoint("Docking Analysis")
    def analyze(self, docking_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the docking results.
        For demonstration, calculate the best score and average score.
        """
        try:
            if not docking_results:
                return {"best_score": None, "average_score": None}

            scores = [result['score'] for result in docking_results if 'score' in result]
            if not scores:
                return {"best_score": None, "average_score": None}

            best_score = min(scores)  # Lower score is better
            average_score = sum(scores) / len(scores)

            analysis = {
                "best_score": best_score,
                "average_score": average_score,
                "total_docked_proteins": len(docking_results)
            }

            logger.info(f"Docking Analysis: {analysis}")
            print(Fore.GREEN + f"Docking Analysis: {analysis}")
            return analysis

        except Exception as e:
            logger.error(f"Error in docking analysis: {e}")
            print(Fore.RED + f"Error in docking analysis: {e}")
            return {}


class SMILESLigandPipeline:
    """Main pipeline for generating, optimizing, and analyzing ligands based on SMILES strings."""

    def __init__(self):
        self.generator = SMILESGenerator()
        self.optimizer = SMILESOptimizer()
        self.predictor = StructurePredictor()
        self.scorer = LigandScorer()
        self.docker = EnsembleDocker()
        self.analyzer = Analyzer()

    def validate_novel_smiles(self, smiles: str) -> bool:
        """Validate that the SMILES string is novel and properly sanitized."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return True
        except:
            return False

    def generate_valid_novel_smiles(self, protein_sequence: str, num_sequences: int) -> List[str]:
        """Generate novel SMILES strings for a given protein sequence."""
        valid_smiles = []
        attempts = 0
        max_attempts = num_sequences * 10  # Allow more attempts for novel molecules
        while len(valid_smiles) < num_sequences and attempts < max_attempts:
            smiles_list = self.generator.generate_smiles_from_protein(protein_sequence, 1)
            if not smiles_list:
                attempts += 1
                continue
            smiles = smiles_list[0]
            if self.validate_novel_smiles(smiles):
                valid_smiles.append(smiles)
            attempts += 1
            if attempts % 10 == 0:
                logger.info(f"Attempted {attempts} times to generate valid SMILES. Found {len(valid_smiles)} valid SMILES so far.")
                print(Fore.YELLOW + f"Attempted {attempts} times to generate valid SMILES. Found {len(valid_smiles)} valid SMILES so far.")
        return valid_smiles

    def generate_protein_structures(self, protein_sequence: str, output_dir: str) -> List[str]:
        """Generate 3D structures for the protein."""
        return self.predictor.predict_3d_protein_structure(protein_sequence, output_dir)

    def score_novel_ligand(self, docking_results: List[Dict[str, Any]], protein_ensemble: List[str]) -> float:
        """Calculate a combined score for the novel SMILES."""
        # Basic physicochemical properties are already considered in the docking score
        docking_score = self.scorer.calculate_docking_score(docking_results, protein_ensemble)
        return docking_score

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
        """
        Main pipeline to generate, optimize, predict structures, dock, analyze, and score ligands.
        """
        all_results = []

        for protein_sequence in protein_sequences:
            try:
                smiles_list = self.generate_valid_novel_smiles(protein_sequence, num_sequences)
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

    def process_single_smiles(self, smiles: str, protein_sequence: str, predicted_structures_dir: str, results_dir: str, score_threshold: float) -> Optional[Dict[str, Any]]:
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
            ligand_pdb = self.predictor.predict_3d_ligand_structure(adjusted_smiles, predicted_structures_dir)
            result['ligand_pdb'] = ligand_pdb
            logger.info(f"3D Structure Prediction completed: {ligand_pdb}")
            print(Fore.GREEN + f"3D Structure Prediction completed: {ligand_pdb}")
        except Exception as e:
            logger.warning(f"Error in 3D structure prediction: {e}")
            print(Fore.YELLOW + f"Error in 3D structure prediction: {e}")
            return None

        # 4. Generate Protein Structures
        try:
            protein_ensemble = self.generate_protein_structures(protein_sequence, predicted_structures_dir)
            logger.info(f"Generated {len(protein_ensemble)} protein structures")
            print(Fore.GREEN + f"Generated {len(protein_ensemble)} protein structures")
        except Exception as e:
            logger.warning(f"Error in protein structure generation: {e}")
            print(Fore.YELLOW + f"Error in protein structure generation: {e}")
            return None

        # 5. Ensemble Docking
        if ligand_pdb and protein_ensemble:
            try:
                docking_results = self.docker.dock_ensemble(ligand_pdb, protein_ensemble, results_dir)
                result['docking_results'] = docking_results
                logger.info("Ensemble Docking completed")
                print(Fore.GREEN + "Ensemble Docking completed")
            except Exception as e:
                logger.warning(f"Error in ensemble docking: {e}")
                print(Fore.YELLOW + f"Error in ensemble docking: {e}")
                return None

        # 6. Analyze Docking Results
        if 'docking_results' in result:
            try:
                analysis = self.analyzer.analyze(result['docking_results'])
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

    def adjust_logp(self, smiles: str, target_logp=2.5, tolerance=0.5) -> str:
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
                mol = self.add_group(mol, 'C')
            else:
                # Add a hydrophilic group (e.g., hydroxyl)
                mol = self.add_group(mol, 'O')

            current_logp = Crippen.MolLogP(mol)

        return Chem.MolToSmiles(mol)

    def add_group(self, mol: Chem.Mol, group: str) -> Chem.Mol:
        """Add a specified group to the molecule."""
        try:
            rwmol = Chem.RWMol(mol)
            group_mol = Chem.MolFromSmiles(group)
            if group_mol is None:
                raise ValueError(f"Invalid group SMILES: {group}")

            combined_mol = Chem.CombineMols(rwmol, group_mol)
            rwmol = Chem.RWMol(combined_mol)

            # Connect the new group to a random atom
            if rwmol.GetNumAtoms() > 1:
                bond_idx = random.randint(0, rwmol.GetNumAtoms() - 2)
                rwmol.AddBond(rwmol.GetNumAtoms() - 1, bond_idx, Chem.BondType.SINGLE)

            new_mol = rwmol.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol
        except Exception as e:
            logger.warning(f"Failed to add group {group} to molecule: {e}")
            print(Fore.YELLOW + f"Failed to add group {group} to molecule: {e}")
            return mol


class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search."""

    def __init__(self, smiles, parent=None):
        self.smiles = smiles
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def is_fully_expanded(self):
        """Check if the node is fully expanded."""
        # Assuming each node can have up to 5 children
        return len(self.children) >= 5

    def best_child(self, c_param=1.4):
        """Select the best child based on UCB1."""
        choices_weights = [
            (child.reward / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]





class SMILESLigandPipeline:
    """Main pipeline for generating, optimizing, and analyzing ligands based on SMILES strings."""

    def __init__(self):
        self.generator = SMILESGenerator()
        self.optimizer = SMILESOptimizer()
        self.predictor = StructurePredictor()
        self.scorer = LigandScorer()
        self.docker = EnsembleDocker()
        self.analyzer = Analyzer()

    def validate_novel_smiles(self, smiles: str) -> bool:
        """Validate that the SMILES string is novel and properly sanitized."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return True
        except:
            return False

    def generate_valid_novel_smiles(self, protein_sequence: str, num_sequences: int) -> List[str]:
        """Generate novel SMILES strings for a given protein sequence."""
        valid_smiles = []
        attempts = 0
        max_attempts = num_sequences * 10  # Allow more attempts for novel molecules
        while len(valid_smiles) < num_sequences and attempts < max_attempts:
            smiles_list = self.generator.generate_smiles_from_protein(protein_sequence, 1)
            if not smiles_list:
                attempts += 1
                continue
            smiles = smiles_list[0]
            if self.validate_novel_smiles(smiles):
                valid_smiles.append(smiles)
            attempts += 1
            if attempts % 10 == 0:
                logger.info(f"Attempted {attempts} times to generate valid SMILES. Found {len(valid_smiles)} valid SMILES so far.")
                print(Fore.YELLOW + f"Attempted {attempts} times to generate valid SMILES. Found {len(valid_smiles)} valid SMILES so far.")
        return valid_smiles

    def generate_protein_structures(self, protein_sequence: str, output_dir: str) -> List[str]:
        """Generate 3D structures for the protein."""
        return self.predictor.predict_3d_protein_structure(protein_sequence, output_dir)

    def score_novel_ligand(self, docking_results: List[Dict[str, Any]], protein_ensemble: List[str]) -> float:
        """Calculate a combined score for the novel SMILES."""
        # Basic physicochemical properties are already considered in the docking score
        docking_score = self.scorer.calculate_docking_score(docking_results, protein_ensemble)
        return docking_score

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
        """
        Main pipeline to generate, optimize, predict structures, dock, analyze, and score ligands.
        """
        all_results = []

        for protein_sequence in protein_sequences:
            try:
                smiles_list = self.generate_valid_novel_smiles(protein_sequence, num_sequences)
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

    def process_single_smiles(self, smiles: str, protein_sequence: str, predicted_structures_dir: str, results_dir: str, score_threshold: float) -> Optional[Dict[str, Any]]:
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
            ligand_pdb = self.predictor.predict_3d_ligand_structure(adjusted_smiles, predicted_structures_dir)
            result['ligand_pdb'] = ligand_pdb
            logger.info(f"3D Structure Prediction completed: {ligand_pdb}")
            print(Fore.GREEN + f"3D Structure Prediction completed: {ligand_pdb}")
        except Exception as e:
            logger.warning(f"Error in 3D structure prediction: {e}")
            print(Fore.YELLOW + f"Error in 3D structure prediction: {e}")
            return None

        # 4. Generate Protein Structures
        try:
            protein_ensemble = self.generate_protein_structures(protein_sequence, predicted_structures_dir)
            logger.info(f"Generated {len(protein_ensemble)} protein structures")
            print(Fore.GREEN + f"Generated {len(protein_ensemble)} protein structures")
        except Exception as e:
            logger.warning(f"Error in protein structure generation: {e}")
            print(Fore.YELLOW + f"Error in protein structure generation: {e}")
            return None

        # 5. Ensemble Docking
        if ligand_pdb and protein_ensemble:
            try:
                docking_results = self.docker.dock_ensemble(ligand_pdb, protein_ensemble, results_dir)
                result['docking_results'] = docking_results
                logger.info("Ensemble Docking completed")
                print(Fore.GREEN + "Ensemble Docking completed")
            except Exception as e:
                logger.warning(f"Error in ensemble docking: {e}")
                print(Fore.YELLOW + f"Error in ensemble docking: {e}")
                return None

        # 6. Analyze Docking Results
        if 'docking_results' in result:
            try:
                analysis = self.analyzer.analyze(result['docking_results'])
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

    def adjust_logp(self, smiles: str, target_logp=2.5, tolerance=0.5) -> str:
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
                mol = self.add_group(mol, 'C')
            else:
                # Add a hydrophilic group (e.g., hydroxyl)
                mol = self.add_group(mol, 'O')

            current_logp = Crippen.MolLogP(mol)

        return Chem.MolToSmiles(mol)

    def add_group(self, mol: Chem.Mol, group: str) -> Chem.Mol:
        """Add a specified group to the molecule."""
        try:
            rwmol = Chem.RWMol(mol)
            group_mol = Chem.MolFromSmiles(group)
            if group_mol is None:
                raise ValueError(f"Invalid group SMILES: {group}")

            combined_mol = Chem.CombineMols(rwmol, group_mol)
            rwmol = Chem.RWMol(combined_mol)

            # Connect the new group to a random atom
            if rwmol.GetNumAtoms() > 1:
                bond_idx = random.randint(0, rwmol.GetNumAtoms() - 2)
                rwmol.AddBond(rwmol.GetNumAtoms() - 1, bond_idx, Chem.BondType.SINGLE)

            new_mol = rwmol.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol
        except Exception as e:
            logger.warning(f"Failed to add group {group} to molecule: {e}")
            print(Fore.YELLOW + f"Failed to add group {group} to molecule: {e}")
            return mol


def main():
    # Example usage of the pipeline
    pipeline = SMILESLigandPipeline()

    predicted_structures_dir = "predicted_structures"
    results_dir = "docking_results"
    num_sequences = 5
    optimization_steps = 10
    score_threshold = -7.0  # Example threshold
    protein_sequences = [
        "MKTIIALSYIFCLVFADYKDDDDK",  # Example protein sequence
        # Add more protein sequences as needed
    ]

    results = pipeline.run_smiles_ligand_pipeline(
        predicted_structures_dir=predicted_structures_dir,
        results_dir=results_dir,
        num_sequences=num_sequences,
        optimization_steps=optimization_steps,
        score_threshold=score_threshold,
        protein_sequences=protein_sequences
    )

    # Display the results
    for res in results:
        print(Style.BRIGHT + Fore.BLUE + "\nResult:")
        for key, value in res.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
