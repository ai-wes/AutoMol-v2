import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy.stats import boltzmann

import torch
from transformers import BertForMaskedLM, BertTokenizer
import asyncio
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Phase_2_protein.predict import predict_protein_function, predict_properties, predict_structure
import random
import numpy as np
from scipy.stats import boltzmann

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

# Monte Carlo approach with simulated annealing
async def monte_carlo_optimize(sequence, iterations=100, initial_temperature=1.0, cooling_rate=0.95):
    current_sequence = sequence
    current_score = await predict_protein_function(current_sequence)
    best_sequence = current_sequence
    best_score = current_score
    temperature = initial_temperature

    for i in range(iterations):
        # Make a random mutation
        mutated_sequence = list(current_sequence)
        mutation_pos = random.randint(0, len(mutated_sequence) - 1)
        mutated_sequence[mutation_pos] = random.choice('ACDEFGHIKLMNPQRSTVWY')
        mutated_sequence = ''.join(mutated_sequence)

        # Evaluate the mutated sequence
        mutated_score = await predict_protein_function(mutated_sequence)

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

        logger.info(f"Iteration {i+1}: Score {current_score}")

    return best_sequence, best_score


# Gradient-based optimization
async def gradient_optimize(sequence, iterations=50, learning_rate=0.01):
    # This is a simplified version. In practice, you'd need a differentiable model for protein function prediction.
    current_sequence = sequence
    for i in range(iterations):
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
        score = await predict_protein_function(current_sequence)
        logger.info(f"Iteration {i+1}: Score {score}")

    return current_sequence, score

# Domain-specific knowledge optimization
async def domain_knowledge_optimize(sequence, iterations=50):
    current_sequence = sequence
    current_score = await predict_protein_function(current_sequence)

    for i in range(iterations):
        # Apply domain-specific rules (this is a simplified example)
        new_sequence = list(current_sequence)

        # Rule 1: Ensure hydrophobic core
        if 'W' not in new_sequence and 'F' not in new_sequence:
            new_sequence[random.randint(0, len(new_sequence)-1)] = random.choice(['W', 'F'])

        # Rule 2: Ensure some charged residues
        if 'K' not in new_sequence and 'R' not in new_sequence and 'D' not in new_sequence and 'E' not in new_sequence:
            new_sequence[random.randint(0, len(new_sequence)-1)] = random.choice(['K', 'R', 'D', 'E'])

        new_sequence = ''.join(new_sequence)
        new_score = await predict_protein_function(new_sequence)

        if new_score > current_score:
            current_sequence = new_sequence
            current_score = new_score

        logger.info(f"Iteration {i+1}: Score {current_score}")

    return current_sequence, current_score

# Ensemble approach
async def ensemble_optimize(sequence, iterations=50):
    methods = [monte_carlo_optimize, gradient_optimize, domain_knowledge_optimize]
    sequences = []
    scores = []

    for method in methods:
        optimized_seq, score = await method(sequence, iterations=iterations//len(methods))
        sequences.append(optimized_seq)
        scores.append(score)

    # Choose the best sequence
    best_index = np.argmax(scores)
    return sequences[best_index], scores[best_index]

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

async def rl_optimize(sequence, iterations=1000, epsilon=0.1, alpha=0.1, gamma=0.9):
    agent = RLAgent(20)  # 20 possible actions (amino acids)
    current_sequence = sequence
    current_score = await predict_protein_function(current_sequence)
    best_sequence = current_sequence
    best_score = current_score

    for i in range(iterations):
        position = random.randint(0, len(current_sequence) - 1)
        current_aa = 'ACDEFGHIKLMNPQRSTVWY'.index(current_sequence[position])
        action = agent.choose_action(current_aa, epsilon)

        new_sequence = list(current_sequence)
        new_sequence[position] = 'ACDEFGHIKLMNPQRSTVWY'[action]
        new_sequence = ''.join(new_sequence)

        new_score = await predict_protein_function(new_sequence)
        reward = new_score - current_score

        agent.update_q_table(current_aa, action, reward, action, alpha, gamma)

        if new_score > current_score:
            current_sequence = new_sequence
            current_score = new_score

            if current_score > best_score:
                best_sequence = current_sequence
                best_score = current_score

        logger.info(f"Iteration {i+1}: Score {current_score}")

    return best_sequence, best_score

async def run_optimization_pipeline(sequences, iterations=50, score_threshold=0.4):
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
            mc_sequence, mc_score = await monte_carlo_optimize(sequence, iterations)
            grad_sequence, grad_score = await gradient_optimize(sequence, iterations)
            dk_sequence, dk_score = await domain_knowledge_optimize(sequence, iterations)
            ensemble_sequence, ensemble_score = await ensemble_optimize(sequence, iterations)
            rl_sequence, rl_score = await rl_optimize(sequence, iterations * 20)  # RL typically needs more iterations

            # Choose the best result
            optimized_sequences = [mc_sequence, grad_sequence, dk_sequence, ensemble_sequence, rl_sequence]
            optimized_scores = [mc_score, grad_score, dk_score, ensemble_score, rl_score]
            best_index = np.argmax(optimized_scores)
            optimized_sequence = optimized_sequences[best_index]
            optimized_score = optimized_scores[best_index]

            if optimized_score >= score_threshold:
                # Predict properties for the final optimized sequence
                properties = await predict_properties(optimized_sequence)
                optimized_results.append({
                    "original_sequence": sequence,
                    "optimized_sequence": optimized_sequence,
                    "original_score": await predict_protein_function(sequence),
                    "optimized_score": optimized_score,
                    "properties": properties
                })
                logger.info(f"Sequence optimization successful. Final score: {optimized_score}")
            else:
                logger.info(f"Sequence optimization did not meet threshold. Score: {optimized_score}")
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            continue

    return optimized_results

if __name__ == "__main__":
    async def main():
        test_sequences = [
            'ITASAWWRSANRSQQLKWTLLGFTCNMVFFPTAHKVQAHATKWLMAREFYGDFNDLTQRAIGPSGGLADHYPTWGYRLMDATGAPGTTMAFLVASLAVFGALVYVIFVVCFAPMAVKDYVAERKVGPIELMMFDVVTLHLLVPFPLLNAASIIAGVQAGIESWGIVSLGVKVGRFGARIPIGVVAAVRLTWMIPRRPAWSADRMRPEPPGPKVYAYRLFSERPIAFDAMFGAVALLGWLTVWRSRKGRVWPSWMGS'
        ]

        results = await run_optimization_pipeline(test_sequences, iterations=50, score_threshold=0.4)

        for result in results:
            print("\nOptimization Result:")
            print(f"Original Sequence: {result['original_sequence'][:50]}...")
            print(f"Optimized Sequence: {result['optimized_sequence'][:50]}...")
            print(f"Original Score: {result['original_score']}")
            print(f"Optimized Score: {result['optimized_score']}")
            print("Properties:")
            for prop, value in result['properties'].items():
                print(f"  {prop}: {value}")

    asyncio.run(main())