import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import os


# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now import predict from the same directory
from phase2a.predict import predict_protein_function, predict_properties

import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
from deap import base, creator, tools, algorithms
import numpy as np
from scipy.stats import boltzmann
import torch
from transformers import BertForMaskedLM, BertTokenizer
from bayes_opt import BayesianOptimization  # New import for Bayesian Optimization
from pyswarm import pso  # New import for Particle Swarm Optimization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load ProtBERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert").to(device)
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

# Define a dictionary of amino acids with known functions or features
aa_functions = {
    'C': 'disulfide bond formation',
    'D': 'negative charge',
    'E': 'negative charge',
    'K': 'positive charge',
    'R': 'positive charge',
    'H': 'metal binding',
    'S': 'phosphorylation site',
    'T': 'phosphorylation site',
    'Y': 'phosphorylation site',
    'W': 'hydrophobic core',
    'F': 'hydrophobic core',
    'L': 'hydrophobic core',
    'I': 'hydrophobic core',
    'V': 'hydrophobic core',
    'G': 'flexibility',
    'P': 'turn formation'
}



##############################################################################################################
############################################################################################################

# Monte Carlo approach with simulated annealing
def monte_carlo_optimize(sequence, iterations=100, initial_temperature=1.0, cooling_rate=0.95):
    current_sequence = sequence
    try:
        current_score = predict_protein_function(current_sequence)
    except Exception as e:
        logger.error(f"Failed to predict protein function for the initial sequence: {e}")
        return current_sequence, 0.0
    best_sequence = current_sequence
    best_score = current_score
    temperature = initial_temperature

    for i in range(iterations):
        try:
            # Make a random mutation
            mutated_sequence = list(current_sequence)
            mutation_pos = random.randint(0, len(mutated_sequence) - 1)
            mutated_sequence[mutation_pos] = random.choice('ACDEFGHIKLMNPQRSTVWY')
            mutated_sequence = ''.join(mutated_sequence)

            # Evaluate the mutated sequence
            mutated_score = predict_protein_function(mutated_sequence)

            # Decide whether to accept the new sequence
            delta_score = mutated_score - current_score
            if delta_score > 0 or random.random() < np.exp(delta_score / temperature):
                current_sequence = mutated_sequence
                current_score = mutated_score

                if current_score > best_score:
                    best_sequence = current_sequence
                    best_score = current_score

            # Cool down the temperature
            temperature *= cooling_rate

            logger.info(f"Monte Carlo Iteration {i+1}: Score {current_score}")
        except Exception as e:
            logger.error(f"Error in Monte Carlo iteration {i+1}: {e}")
            continue

    return best_sequence, best_score

##############################################################################################################
############################################################################################################

# Gradient-based optimization
def gradient_optimize(sequence, iterations=50, learning_rate=0.01):
    # This is a simplified version. In practice, you'd need a differentiable model for protein function prediction.
    current_sequence = sequence
    try:
        current_score = predict_protein_function(current_sequence)
    except Exception as e:
        logger.error(f"Failed to predict protein function for the initial sequence: {e}")
        return current_sequence, 0.0

    for i in range(iterations):
        try:
            # Compute "gradient" (this is a placeholder for actual gradient computation)
            gradient = np.random.randn(len(current_sequence))

            # Update sequence based on gradient
            new_sequence = ''
            for j, aa in enumerate(current_sequence):
                aa_index = 'ACDEFGHIKLMNPQRSTVWY'.index(aa)
                new_index = int((aa_index + learning_rate * gradient[j]) % 20)
                new_sequence += 'ACDEFGHIKLMNPQRSTVWY'[new_index]

            current_sequence = new_sequence

            # Evaluate new sequence
            score = predict_protein_function(current_sequence)
            logger.info(f"Gradient Iteration {i+1}: Score {score}")
        except Exception as e:
            logger.error(f"Error in Gradient iteration {i+1}: {e}")
            continue

    return current_sequence, score



##############################################################################################################
############################################################################################################

# Domain-specific knowledge optimization
def domain_knowledge_optimize(sequence, iterations=50):
    current_sequence = sequence
    try:
        current_score = predict_protein_function(current_sequence)
    except Exception as e:
        logger.error(f"Failed to predict protein function for the initial sequence: {e}")
        return current_sequence, 0.0

    for i in range(iterations):
        try:
            # Apply domain-specific rules (this is a simplified example)
            new_sequence = list(current_sequence)

            # Rule 1: Ensure hydrophobic core
            if 'W' not in new_sequence and 'F' not in new_sequence:
                new_sequence[random.randint(0, len(new_sequence)-1)] = random.choice(['W', 'F'])

            # Rule 2: Ensure some charged residues
            if 'K' not in new_sequence and 'R' not in new_sequence and 'D' not in new_sequence and 'E' not in new_sequence:
                new_sequence[random.randint(0, len(new_sequence)-1)] = random.choice(['K', 'R', 'D', 'E'])

            new_sequence = ''.join(new_sequence)
            new_score = predict_protein_function(new_sequence)

            if new_score > current_score:
                current_sequence = new_sequence
                current_score = new_score

            logger.info(f"Domain Knowledge Iteration {i+1}: Score {current_score}")
        except Exception as e:
            logger.error(f"Error in Domain Knowledge iteration {i+1}: {e}")
            continue

    return current_sequence, current_score


##############################################################################################################
############################################################################################################

# Ensemble approach
def ensemble_optimize(sequence, iterations=50):
    methods = [monte_carlo_optimize, gradient_optimize, domain_knowledge_optimize]
    sequences = []
    scores = []

    for method in methods:
        try:
            optimized_seq, score = method(sequence, iterations=iterations//len(methods))
            sequences.append(optimized_seq)
            scores.append(score)
        except Exception as e:
            logger.error(f"Error in ensemble method {method.__name__}: {e}")
            continue

    if not scores:
        logger.warning("No successful optimization methods in ensemble.")
        return sequence, 0.0

    # Choose the best sequence
    best_index = np.argmax(scores)
    return sequences[best_index], scores[best_index]


##############################################################################################################
############################################################################################################


# Simplified Reinforcement Learning
class RLAgent:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        self.q_table = np.zeros((20, action_space_size))  # 20 amino acids

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        current_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
        self.q_table[state, action] = new_q


def rl_optimize(sequence, iterations=1000, epsilon=0.1, alpha=0.1, gamma=0.9):
    agent = RLAgent(20)  # 20 possible actions (amino acids)
    current_sequence = sequence
    try:
        current_score = predict_protein_function(current_sequence)
    except Exception as e:
        logger.error(f"Failed to predict protein function for the initial sequence: {e}")
        return current_sequence, 0.0
    best_sequence = current_sequence
    best_score = current_score

    for i in range(iterations):
        try:
            position = random.randint(0, len(current_sequence) - 1)
            current_aa = 'ACDEFGHIKLMNPQRSTVWY'.index(current_sequence[position])
            action = agent.choose_action(current_aa, epsilon)

            new_sequence = list(current_sequence)
            new_sequence[position] = 'ACDEFGHIKLMNPQRSTVWY'[action]
            new_sequence = ''.join(new_sequence)

            new_score = predict_protein_function(new_sequence)
            reward = new_score - current_score

            agent.update_q_table(current_aa, action, reward, action, alpha, gamma)

            if new_score > current_score:
                current_sequence = new_sequence
                current_score = new_score

                if current_score > best_score:
                    best_sequence = current_sequence
                    best_score = current_score

            logger.info(f"Reinforcement Learning Iteration {i+1}: Score {current_score}")
        except Exception as e:
            logger.error(f"Error in Reinforcement Learning iteration {i+1}: {e}")
            continue

    return best_sequence, best_score


##############################################################################################################
############################################################################################################


# Constraint-based optimization
def constraint_optimize(sequence, iterations=50, min_hydrophobicity=0.3, max_charge=0.2):
    current_sequence = sequence
    try:
        current_score = predict_protein_function(current_sequence)
    except Exception as e:
        logger.error(f"Failed to predict protein function for the initial sequence: {e}")
        return current_sequence, 0.0
    best_sequence = current_sequence
    best_score = current_score

    for i in range(iterations):
        try:
            # Make a random mutation
            mutated_sequence = list(current_sequence)
            mutation_pos = random.randint(0, len(mutated_sequence) - 1)
            mutated_sequence[mutation_pos] = random.choice('ACDEFGHIKLMNPQRSTVWY')
            mutated_sequence = ''.join(mutated_sequence)

            # Check constraints
            analysis = ProteinAnalysis(mutated_sequence)
            hydrophobicity = analysis.gravy()
            charge = abs(analysis.charge_at_pH(7.0)) / len(mutated_sequence)

            if hydrophobicity >= min_hydrophobicity and charge <= max_charge:
                # Evaluate the mutated sequence
                mutated_score = predict_protein_function(mutated_sequence)

                # Decide whether to accept the new sequence
                delta_score = mutated_score - current_score
                if delta_score > 0 or random.random() < np.exp(delta_score / temperature):
                    current_sequence = mutated_sequence
                    current_score = mutated_score

                    if current_score > best_score:
                        best_sequence = current_sequence
                        best_score = current_score

            # Cool down the temperature
            temperature *= cooling_rate

            logger.info(f"Constraint Iteration {i+1}: Score {current_score}")
        except Exception as e:
            logger.error(f"Error in Constraint iteration {i+1}: {e}")
            continue

    return best_sequence, best_score


##############################################################################################################
############################################################################################################


##############################################################################################################
############################################################################################################
## Diversity preservation
def hamming_distance(seq1, seq2):
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
def diversity_optimize(sequence, population_size=50, generations=50):
    def create_individual():
        return ''.join(random.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(len(sequence)))

    population = [create_individual() for _ in range(population_size)]

    for gen in range(generations):
        try:
            # Evaluate fitness
            fitness_scores = [predict_protein_function(ind) for ind in population]

            # Sort population by fitness
            sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)

            # Select top individuals
            elite_size = population_size // 4
            new_population = [ind for ind, _ in sorted_population[:elite_size]]

            # Create new individuals
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(population, 2)
                child = ''.join(p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2))

                # Mutation
                if random.random() < 0.1:
                    pos = random.randint(0, len(child) - 1)
                    child = child[:pos] + random.choice('ACDEFGHIKLMNPQRSTVWY') + child[pos+1:]

                # Add to population if diverse enough
                if all(hamming_distance(child, ind) > len(sequence) // 10 for ind in new_population):
                    new_population.append(child)

            population = new_population

            logger.info(f"Diversity Generation {gen+1}: Best score {sorted_population[0][1]}")
        except Exception as e:
            logger.error(f"Error in Diversity Generation {gen+1}: {e}")
            continue

    best_sequence = sorted_population[0][0]
    best_score = sorted_population[0][1]
    return best_sequence, best_score

##############################################################################################################
############################################################################################################


# Constrained Monte Carlo optimization
def constrained_monte_carlo_optimize(sequence, iterations=100, initial_temperature=1.0, cooling_rate=0.95, min_hydrophobicity=0.3, max_charge=0.2):
    current_sequence = sequence
    try:
        current_score = predict_protein_function(current_sequence)
    except Exception as e:
        logger.error(f"Failed to predict protein function for the initial sequence: {e}")
        return current_sequence, 0.0
    best_sequence = current_sequence
    best_score = current_score
    temperature = initial_temperature

    for i in range(iterations):
        try:
            # Make a random mutation
            mutated_sequence = list(current_sequence)
            mutation_pos = random.randint(0, len(mutated_sequence) - 1)
            mutated_sequence[mutation_pos] = random.choice('ACDEFGHIKLMNPQRSTVWY')
            mutated_sequence = ''.join(mutated_sequence)

            # Check constraints
            analysis = ProteinAnalysis(mutated_sequence)
            hydrophobicity = analysis.gravy()
            charge = abs(analysis.charge_at_pH(7.0)) / len(mutated_sequence)

            if hydrophobicity >= min_hydrophobicity and charge <= max_charge:
                # Evaluate the mutated sequence
                mutated_score = predict_protein_function(mutated_sequence)

                # Decide whether to accept the new sequence
                delta_score = mutated_score - current_score
                if delta_score > 0 or random.random() < np.exp(delta_score / temperature):
                    current_sequence = mutated_sequence
                    current_score = mutated_score

                    if current_score > best_score:
                        best_sequence = current_sequence
                        best_score = current_score

            # Cool down the temperature
            temperature *= cooling_rate

            logger.info(f"Constrained Monte Carlo Iteration {i+1}: Score {current_score}")
        except Exception as e:
            logger.error(f"Error in Constrained Monte Carlo iteration {i+1}: {e}")
            continue

    return best_sequence, best_score


##############################################################################################################
############################################################################################################

# Adaptive Monte Carlo optimization
def adaptive_monte_carlo_optimize(sequence, iterations=100, initial_temperature=1.0, cooling_rate=0.95, adaptation_rate=0.1):
    current_sequence = sequence
    try:
        current_score = predict_protein_function(current_sequence)
    except Exception as e:
        logger.error(f"Failed to predict protein function for the initial sequence: {e}")
        return current_sequence, 0.0
    best_sequence = current_sequence
    best_score = current_score
    temperature = initial_temperature
    mutation_probabilities = {aa: 1/20 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    for i in range(iterations):
        try:
            # Make a random mutation based on probabilities
            mutated_sequence = list(current_sequence)
            mutation_pos = random.randint(0, len(mutated_sequence) - 1)
            new_aa = random.choices('ACDEFGHIKLMNPQRSTVWY', weights=[mutation_probabilities[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'])[0]
            mutated_sequence[mutation_pos] = new_aa
            mutated_sequence = ''.join(mutated_sequence)

            # Evaluate the mutated sequence
            mutated_score = predict_protein_function(mutated_sequence)

            # Decide whether to accept the new sequence
            delta_score = mutated_score - current_score
            if delta_score > 0 or random.random() < np.exp(delta_score / temperature):
                current_sequence = mutated_sequence
                current_score = mutated_score

                if current_score > best_score:
                    best_sequence = current_sequence
                    best_score = current_score

                # Update mutation probabilities
                for aa in 'ACDEFGHIKLMNPQRSTVWY':
                    if aa == new_aa:
                        mutation_probabilities[aa] += adaptation_rate
                    else:
                        mutation_probabilities[aa] = max(0, mutation_probabilities[aa] - adaptation_rate / 19)

                # Normalize probabilities
                total = sum(mutation_probabilities.values())
                for aa in mutation_probabilities:
                    mutation_probabilities[aa] /= total

            # Cool down the temperature
            temperature *= cooling_rate

            logger.info(f"Adaptive Monte Carlo Iteration {i+1}: Score {current_score}")
        except Exception as e:
            logger.error(f"Error in Adaptive Monte Carlo iteration {i+1}: {e}")
            continue

    return best_sequence, best_score


##############################################################################################################
############################################################################################################

# Bayesian Optimization approach
def bayesian_optimize(sequence, iterations=30):
    try:
        current_score = predict_protein_function(sequence)
    except Exception as e:
        logger.error(f"Failed to predict protein function for the initial sequence: {e}")
        return sequence, 0.0

    def evaluate(sequence_numeric):
        """
        Convert numerical vector back to amino acid sequence and evaluate.
        """
        try:
            sequence_str = ''.join([ 'ACDEFGHIKLMNPQRSTVWY'[int(round(num)) % 20] for num in sequence_numeric ])
            score = predict_protein_function(sequence_str)
            return score
        except Exception as e:
            logger.error(f"Error in Bayesian Optimization evaluation: {e}")
            return 0.0

    # Define the bounds for each position as integers between 0 and 19
    bounds = [(0, 19) for _ in range(len(sequence))]

    optimizer = BayesianOptimization(
        f=lambda *args: evaluate(args),
        pbounds={f'x{i}': (0, 19) for i in range(len(sequence))},
        verbose=2
    )

    try:
        optimizer.maximize(
            init_points=5,
            n_iter=iterations
        )
    except Exception as e:
        logger.error(f"Error during Bayesian Optimization: {e}")
        return sequence, current_score

    # Extract the best sequence found
    try:
        best_params = optimizer.max['params']
        best_sequence = ''.join([ 'ACDEFGHIKLMNPQRSTVWY'[int(round(best_params[f'x{i}'])) % 20] for i in range(len(sequence)) ])
        best_score = optimizer.max['target']
        return best_sequence, best_score
    except Exception as e:
        logger.error(f"Error extracting best parameters from Bayesian Optimization: {e}")
        return sequence, current_score


##############################################################################################################
############################################################################################################

# Particle Swarm Optimization approach
# Particle Swarm Optimization approach
def pso_optimize(sequence, iterations=50):
    try:
        current_score = predict_protein_function(sequence)
    except Exception as e:
        logger.error(f"Failed to predict protein function for the initial sequence: {e}")
        return sequence, 0.0

    def evaluate(sequence_numeric):
        """
        Convert numerical vector back to amino acid sequence and evaluate.
        """
        try:
            sequence_str = ''.join([ 'ACDEFGHIKLMNPQRSTVWY'[int(round(num)) % 20] for num in sequence_numeric ])
            score = predict_protein_function(sequence_str)
            return -score  # Minimize negative score
        except Exception as e:
            logger.error(f"Error in PSO evaluation: {e}")
            return float('inf')

    # Define the bounds for each position as integers between 0 and 19
    lb = [0] * len(sequence)
    ub = [19] * len(sequence)

    try:
        from pyswarm import pso as psopso
    except ImportError:
        logger.error("pyswarm library is not installed. Please install it using 'pip install pyswarm'")
        return sequence, current_score

    try:
        best_sequence_numeric, best_score = psopso(evaluate, lb, ub, swarmsize=30, maxiter=iterations, debug=True)
        best_sequence = ''.join([ 'ACDEFGHIKLMNPQRSTVWY'[int(round(num)) % 20] for num in best_sequence_numeric ])
        best_score = -best_score
        return best_sequence, best_score
    except Exception as e:
        logger.error(f"Error during Particle Swarm Optimization: {e}")
        return sequence, current_score

##############################################################################################################
############################################################################################################


def run_optimization_pipeline(sequences, iterations=50, score_threshold=0.4):
    optimized_results = []
    print(f"Starting optimization pipeline with sequences: {sequences}")
    for sequence in sequences:
        try:
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            sequence = ''.join(char for char in sequence if char in valid_aa)
            if not sequence:
                logger.warning("Skipping empty or invalid sequence")
                continue

            # Run all optimization methods
            monte_carlo_sequence, monte_carlo_score = monte_carlo_optimize(sequence, iterations)
            gradient_sequence, gradient_score = gradient_optimize(sequence, iterations)
            domain_knowledge_sequence, domain_knowledge_score = domain_knowledge_optimize(sequence, iterations)
            ensemble_sequence, ensemble_score = ensemble_optimize(sequence, iterations)
            rl_sequence, rl_score = rl_optimize(sequence, iterations)
            constraint_sequence, constraint_score = constraint_optimize(sequence, iterations)
            diversity_sequence, diversity_score = diversity_optimize(sequence, generations=iterations)
            constrained_mc_sequence, constrained_mc_score = constrained_monte_carlo_optimize(sequence, iterations)
            adaptive_mc_sequence, adaptive_mc_score = adaptive_monte_carlo_optimize(sequence, iterations)
            
            # New methods
            bayesian_sequence, bayesian_score = bayesian_optimize(sequence, iterations=30)
            pso_sequence, pso_score = pso_optimize(sequence, iterations=50)

            # Choose the best result
            optimized_sequences = [
                monte_carlo_sequence, gradient_sequence, domain_knowledge_sequence, ensemble_sequence, 
                rl_sequence, constraint_sequence, diversity_sequence, 
                constrained_mc_sequence, adaptive_mc_sequence, bayesian_sequence, pso_sequence
            ]
            optimized_scores = [
                monte_carlo_score, gradient_score, domain_knowledge_score, ensemble_score, 
                rl_score, constraint_score, diversity_score, 
                constrained_mc_score, adaptive_mc_score, bayesian_score, pso_score
            ]
            best_index = np.argmax(optimized_scores)
            optimized_sequence = optimized_sequences[best_index]
            optimized_score = optimized_scores[best_index]

            if optimized_score >= score_threshold:
                try:
                    original_score = predict_protein_function(sequence)
                except Exception as e:
                    logger.error(f"Failed to predict original protein function: {e}")
                    original_score = 0.0

                # Predict properties for the final optimized sequence
                try:
                    properties = predict_properties(optimized_sequence)
                except Exception as e:
                    logger.error(f"Failed to predict properties for the optimized sequence: {e}")
                    properties = {}

                optimized_results.append({
                    "original_sequence": sequence,
                    "optimized_sequence": optimized_sequence,
                    "original_score": original_score,
                    "optimized_score": optimized_score,
                    "properties": properties,
                    "best_method": [
                        "Monte Carlo", "Gradient", "Domain Knowledge", "Ensemble", "Reinforcement Learning", 
                        "Constraint", "Diversity", "Constrained Monte Carlo", "Adaptive Monte Carlo",
                        "Bayesian Optimization", "Particle Swarm Optimization"
                    ][best_index]
                })
                logger.info(f"Sequence optimization successful. Final score: {optimized_score}")
                logger.info(f"Best method: {optimized_results[-1]['best_method']}")
            else:
                logger.info(f"Sequence optimization did not meet threshold. Score: {optimized_score}")
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            continue

    return optimized_results


def main():
    # Example sequences to optimize
    sequences = [
        "HMPYHFQGTNWFGCIASVRNGMRTKWFELSFAWYERSMMYQWNKVWWTKFYWYVWFLWQKLWQQMLYWAVGWRHFHPTRHPSFHVKKFFVQKAVSFAICHPWYKWKHPQQRFIQGRISMQNPWHPMNVNTFHLDWKLIFKYQNLWIGNEWMLITWKDWRFNPYWIWKSKLGIWWWYWYTWFRHMFRNAFQHMVTQYYINNFYRLMMFVLDSFKELITYWFRFRKHGQGRCNWQTAYWFYKYTFDHERRVGPIS",
        "KFRWSWNRYYKWNWQTWWNTQWFVQMWTTIFFYWSFMMGRTWGWRTRYSYAFFFTFMLLYKIHTWAEMWPMIYWTCGAMMYQHFWVFSWWPHKLYIWQRIWRHRKMRYWIHDWWYQCFYKHVSNWQLLREKSTFKGNNFFPWNWAVFRRPCYRPRHWIPERMVVRPYHDNVFFRKLLRTDQKSIRYNSLVRFCIKHLYFKMQNPNGAVPYFNHFWYWYYTFTQWMKIVYWHYWKLCVFDKFWPNYLHMPAQFSAII",
        "FKVFTRVFFWTWKLYEYEFHAHWWTWPGHYYLYWYRVIKEKEWWNNFTLFWRMWGWFPLRRWKPPWKSWLYWWTQQMPCSFILFWRFSFRAAMMWHMRGENRARLKPIMTYWQVWDQVWEAWRIRPWYKQCANHDRFWWEPHSPLMRFDWWQIIFFKPRNTSLNFHWRYSKAQYNPLFKWYYLTWWFVMVNWQHVCYFAYQKFGKWCHYWIHLTFWIAMPTNEVLHCRNAHFIRYRMFWWRCITARNNQISHD",
        "LLRTQVRMIPNYAAWYMAHMWWYIVMGRDWFFAWYRQFVLGWPRIRYKPKGLHRYIQHNNKHTWFSQHFTFQCKWWWKKHWPWPRLHMWKWWLYTNYVFMIMDVGATWMNHEKYTLACRHWDWMKKIKWWDNYNTRVWTMQFFPGFFKLTYWIIQNWWHWLAWRYQWFQIVYPFTWWQYWVSQFLENPWPRDAWRHRGSFLPHISRSWFWNTWGGFVRKMVFNMKNIWRFSSVFDRIMQLWPWWTLSGRFSFIPFK",
        "FGTLKVQMYWWNGWKGAWWLYFRTHRRHGKSKYFFFTWWFPKWWNSPRQPMYSWWAPIFCRYSMEGNVMEYNWAKWREYEWDQWFWLEKLIGFEVFNARNHLRNRQWKSNFQPWSLEHPYWLDFFRFGTIMYWWSHWCFWIWFWRMNYTNYQQHFVWTTHPPPWPLQWTQMFQIYWEVAIKFRLMEHRMHFKWWRATPWIVMARKNMGFVSIYKYWRKVRVWVRTNFWHLYFAMRMSFKYLRHHCEWIWRWIHTQA"
    ]

    # Set optimization parameters
    iterations = 100
    score_threshold = 0.8  # Increased threshold for higher optimization

    # Run optimization
    optimized_results = run_optimization_pipeline(sequences, iterations, score_threshold)

    # Print results
    for result in optimized_results:
        print(f"Original sequence: {result['original_sequence']}")
        print(f"Optimized sequence: {result['optimized_sequence']}")
        print(f"Original score: {result['original_score']}")
        print(f"Optimized score: {result['optimized_score']}")
        print(f"Best method: {result['best_method']}")
        print("Properties:")
        for prop, value in result['properties'].items():
            print(f"  {prop}: {value}")
        print("\n")
