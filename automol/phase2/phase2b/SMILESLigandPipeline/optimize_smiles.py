from rdkit.Chem import AllChem
import random
import logging
from colorama import Fore
from rdkit import Chem
from rdkit.Chem import Descriptors
from deap import base, creator, tools, algorithms

logger = logging.getLogger(__name__)

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

class SMILESOptimizer:
    """Optimizes SMILES strings using genetic algorithms."""

    def __init__(self, population_size=200, generations=0):
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
        """Initialize an individual with the original SMILES."""
        return creator.Individual([self.original_smiles])

    @checkpoint("SMILES Optimization")
    def optimize_smiles(self, smiles: str) -> str:
        """Optimize a single SMILES string."""
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
        """Apply mutation to an individual."""
        smiles = individual[0]
        new_smiles = self.mutate_smiles(smiles)
        individual[0] = new_smiles
        return (individual,)

    def validate_smiles(self, smiles: str) -> bool:
        """Validate the SMILES string based on specific criteria."""
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
        
        
        

    @checkpoint("Fragment-Based Optimization")
    def fragment_based_optimization(self, smiles: str) -> str:
        """Optimize the molecule based on predefined drug-like fragments."""
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
    def bioisostere_replacement(self, smiles: str) -> str:
        """Replace functional groups with their bioisosteres."""
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

    @checkpoint("Ring Expansion")
    def ring_expansion(self, smiles: str) -> str:
        """Expand rings in the molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # Check for ring closure
        if not mol.GetRingInfo().IsInitialized():
            Chem.AssignStereochemistryFrom3D(mol)
    
