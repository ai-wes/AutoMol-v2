
import sys

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
from rdkit import DataStructs
import csv
from datetime import datetime

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configuration Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_SIZE = 2048
HIDDEN_SIZE = 256
ACTION_SIZE = 1 + (20 * 21)  # Termination action + (max_atoms * num_fragments)
DROPOUT_P = 0.2
PRETRAINED_MODEL_NAME = 'mrm8488/chEMBL_smiles_v1'
LEARNING_RATE = 1e-4
SAVE_INTERVAL = 1000
FILE_DIR = r'C:\Users\wes\MOL_RL_AGENT_v2\checkpoints'
os.makedirs(FILE_DIR, exist_ok=True)
print(f"File directory: {FILE_DIR}")



# At the beginning of your script, define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Define fragments (functional groups or substructures)
fragments = [
    'C',  # methyl group
    'O',  # oxygen atom
    'N',  # nitrogen atom
    'S',  # sulfur atom
    'Cl',  # chlorine atom
    'F',  # fluorine atom
    'Br',  # bromine atom
    'I',  # iodine atom
    'C=O',  # carbonyl group
    'C#N',  # nitrile group
    'C=C',  # alkene group
    'C#C',  # alkyne group
    'C1CC1',  # cyclopropane ring
    'C1CCC1',  # cyclobutane ring
    'C1CCCC1',  # cyclopentane ring
    'C1CCCCC1',  # cyclohexane ring
    'c1ccccc1',  # benzene ring
    'C(=O)O',  # carboxylic acid group
    'C(=O)N',  # amide group
    'C1=CC=CC=C1',  # another benzene ring
    # More fragments can be added here
]

# Define Aging pathways
AGING_PATHWAYS = [
    "Autophagy Induction",
    "Epigenetic Modulation",
    "Mitochondrial Function Enhancement",
    "Extracellular Matrix Modulation",
    "Stem Cell Niche Enhancement",
    "Senomorphic Effects",
    "Exosome Modulation",
    "Cellular Reprogramming",
    "Telomere Protection",
    "Cellular Senescence Pathway Modulation",
    "mTOR Inhibition",
    "Sirtuin Activation",
    "Senolytic Activity",
    "DNA Repair Enhancement",
    "Proteostasis Enhancement",
    "Circadian Rhythm Regulation",
    "Hormesis Induction",
    "Reprogramming Factor Mimetics"
]

# Define the known active molecules for each pathway
PATHWAY_ACTIVES = {
    "Autophagy Induction": [
        "CC1=CC=C(C=C1)NC(=O)C",
        "COC1=C(C=C(C=C1)NC(=O)C)OC",
        "CC(=O)NC1=CC=C(C=C1)O",
        "CC1=C(C(=CC=C1)C)NC(=O)CC2=CC=CC=C2",
        "COC1=CC(=CC(=C1OC)OC)C(=O)NC2=CC=CC=C2",
    ],
    "Epigenetic Modulation": [
        "CC(=O)NC1=CC=CC=C1",
        "CC1=C(C=C(C=C1)S(=O)(=O)NC2=CC=CC=C2)C",
        "CC1=CC(=NO1)C2=CC(=CC=C2)S(=O)(=O)NC3=CC=CC=C3",
        "CC1=C(C(=O)NC(=O)N1)C2=CC=CC=C2",
        "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    ],
    "Mitochondrial Function Enhancement": [
        "C1=CC(=CC=C1C(=O)O)O",
        "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=O)O)C",
        "CC1=C(C(=O)C2=C(C1=O)C(=CC=C2)O)O",
        "CC1=CC=C(C=C1)OCC(C)(C)NC2=NC=NC3=C2C=CN3",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCOCC3)OC",
    ],
    "Extracellular Matrix Modulation": [  # Added pathway
        "CC(=O)Nc1ccc(cc1)C(=O)O",  # Example SMILES
        "C1=CC=CC=C1C(=O)O",        # Example SMILES
        "CCN(CC)C(=O)C1=CC=CC=C1",  # Example SMILES
        "CC(=O)Nc1ccc(cc1)N",        # Example SMILES
        "CCOCC(=O)Nc1ccc(cc1)O",     # Example SMILES
    ],
    "Stem Cell Niche Enhancement": [
        "CC1=C(C(=O)N2CCCC2=N1)C3=CC=C(C=C3)Cl",
        "CC1=NC(=C(N1CC2=CC=C(C=C2)C(=O)O)C(=O)NC3=CC=CC=C3)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
        "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
    ],
    "Senomorphic Effects": [
        "CC1=C(C=C(C=C1)C(=O)NC2=CC=CC=C2)NC(=O)OC3CCCC3",
        "CC1=CC=C(C=C1)C2=CC(=NO2)C(=O)NC3=CC=CC=C3C(F)(F)F",
        "COC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)C3=CN=C(N=C3N)N",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Exosome Modulation": [  # Added pathway
        "CC(=O)Oc1ccccc1C(=O)O",      # Example SMILES
        "CC(=O)N1CCC(CC1)C(=O)O",     # Example SMILES
        "C1=CC=CC=C1NC(=O)O",        # Example SMILES
        "CC(=O)N(C)C1=CC=CC=C1",      # Example SMILES
        "CCC(=O)Nc1ccc(cc1)O",          # Example SMILES
    ],
    "Cellular Reprogramming": [
        "CC(=O)Nc1ccc(cc1)N",          # Example SMILES
        "CC1=CC=CC=C1NC(=O)O",         # Example SMILES
        "CCC(=O)Nc1ccc(cc1)C(=O)O",    # Example SMILES
        "CC(=O)NCC1=CC=CC=C1",         # Example SMILES
        "CC(=O)Nc1ccc(cc1)C#N",        # Example SMILES
    ],
    "Telomere Protection": [  # Corrected name
        "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C3=CC=CC=C3",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
        "CC1=C(C2=CC=CC=C2N1)C(=O)NC3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
    ],
    "Cellular Senescence Pathway Modulation": [  # Added pathway
        "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
        "C1=CC=C(C=C1)C2=C(C(=O)OC3=CC=CC=C23)O",
        "CCC(=O)Nc1ccccc1O",
        "CC(=O)NCC1=CC=CC=C1",
        "CC(=O)Nc1ccc(cc1)O",
    ],
    "mTOR Inhibition": [
        "CC1=CC=C(C=C1)C2=CN=C(N=C2N)N",  # Rapamycin-like
        "CCC(=O)Nc1ccc(cc1)O",            # Example SMILES
        "CC(=O)NCC1=CC=CC=C1",           # Example SMILES
        "CC(=O)Nc1ccc(cc1)C#N",          # Example SMILES
        "CC(=O)Nc1ccccc1O",              # Example SMILES
    ],
    "Sirtuin Activation": [
        "CC1=CC(=CC=C1O)C=CC(=O)C2=CC=C(C=C2)O",  # Resveratrol-like
        "C1=CC=C(C=C1O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O",  # Fisetin-like
        "CC(=O)Nc1ccc(cc1)O",                       # Example SMILES
        "CC(=O)NCC1=CC=CC=C1",                      # Example SMILES
        "CCC(=O)Nc1ccccc1O",                        # Example SMILES
    ],
    "Senolytic Activity": [
        "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",  # Quercetin-like
        "C1=CC=C(C=C1)C2=C(C(=O)OC3=CC=CC=C23)O",  # Flavonoid-like
        "CCC(=O)Nc1ccc(cc1)O",             # Example SMILES
        "CC(=O)NCC1=CC=CC=C1",              # Example SMILES
        "CC(=O)Nc1ccc(cc1)O",                # Example SMILES
    ],
    "DNA Repair Enhancement": [
        "C1=NC2=C(N1)C(=O)NC(=O)N2",        # NAD+ precursor-like
        "C1=NC(=O)NC(=O)C1",                # Nicotinamide-like
        "CC(=O)Nc1ccc(cc1)O",               # Example SMILES
        "CCC(=O)Nc1ccccc1O",                # Example SMILES
        "CC(=O)NCC1=CC=CC=C1",              # Example SMILES
    ],
    "Proteostasis Enhancement": [
        "CC1=C(C=C(C=C1)C(=O)NC2=CC=CC=C2)C",
        "COC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)C3=CN=C(N=C3N)N",
        "CC1=CC(=NO1)C2=CC(=CC=C2)S(=O)(=O)NC3=CC=CC=C3",
        "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
    ],
    "Circadian Rhythm Regulation": [
        "C1=CC2=C(C=C1)NC(=O)C=C2",          # Melatonin-like
        "C1=CC=C2C(=C1)C(=O)CC(=O)N2",      # Circadian rhythm modulator-like
        "CCC(=O)Nc1ccccc1O",                # Example SMILES
        "CC(=O)NCC1=CC=CC=C1",               # Example SMILES
        "CC(=O)Nc1ccc(cc1)O",                # Example SMILES
    ],
    "Hormesis Induction": [
        "C1=CC=C(C=C1)O",                    # Simple phenol-like
        "C1=CC(=CC=C1O)O",                   # Catechol-like
        "CC(=O)Nc1ccccc1O",                  # Example SMILES
        "CCC(=O)Nc1ccc(cc1)O",               # Example SMILES
        "CC(=O)NCC1=CC=CC=C1",               # Example SMILES
    ],
    "Reprogramming Factor Mimetics": [
        "C1=NC2=C(N1)C(=O)NC(=O)N2",        # Similar to cellular reprogramming
        "C1=NC(=O)NC(=O)C1",                # Alternative reprogramming factor mimetic-like
        "CCC(=O)Nc1ccccc1O",                # Example SMILES
        "CC(=O)NCC1=CC=CC=C1",               # Example SMILES
        "CC(=O)Nc1ccc(cc1)O",                # Example SMILES
    ],
}

# Define transformations and ensure they are correct
TRANSFORMATIONS = [
    ('Add Hydroxyl', '[C:1][H]>>[C:1][OH]'),
    ('Add Amine', '[C:1][H]>>[C:1][NH2]'),
    ('Add Methyl', '[C:1][H]>>[C:1][CH3]'),
    ('Add Carboxyl', '[C:1][H]>>[C:1][C](=O)[OH]'),
    ('Replace OH with NH2', '[C:1][OH]>>[C:1][NH2]'),
    ('Replace CH3 with OH', '[C:1][CH3]>>[C:1][OH]'),
    ('Add Nitro', '[C:1][H]>>[C:1][N+](=O)[O-]'),
    ('Add Sulfhydryl', '[C:1][H]>>[C:1][SH]'),
    ('Add Ether', '[C:1][H]>>[C:1][O][C:2]'),
    ('Form Ring', '[C:1]-[C:2]>>[C:1]1-[C:2]-[C]-[C]-[C]-[C]-1'),
    ('Break Ring', '[C:1]1-[C:2]-[C]-[C]-[C]-[C]-1>>[C:1]-[C:2]'),
    ('Add Benzene Ring', '[C:1][H]>>[C:1]C1=CC=CC=C1'),
    ('Modify Nitrogen', '[N:1]=[C:2]>>[N:1][C:2]'),
    ('Add Halogen', '[C:1][H]>>[C:1][Cl]'),
    ('Add Phenol', '[c:1][H]>>[c:1][OH]'),
    ('Add Thiol', '[C:1][H]>>[C:1][SH]'),
    # Additional transformations can be uncommented and added here
]


REACTIONS = [(name, AllChem.ReactionFromSmarts(smarts)) for name, smarts in TRANSFORMATIONS]

class Policy(nn.Module):
    def __init__(self, pretrained_model_name, hidden_size, action_size, dropout_p, fine_tune=False):
        super(Policy, self).__init__()
        # Example architecture; adjust based on your actual model
        self.fc1 = nn.Linear(STATE_SIZE, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        action_probs = F.softmax(action_scores, dim=-1)
        return action_probs, state_values

class MoleculeEnv:
    def __init__(self, max_steps=10, max_atoms=20, curriculum_level=1):
        """
        Initializes the Molecule Environment for RL agent.
        
        Args:
            max_steps (int): Maximum steps per episode.
            max_atoms (int): Maximum number of atoms in a molecule.
            curriculum_level (int): Current curriculum level.
        """
        self.fragments = [Chem.MolFromSmiles(frag) for frag in fragments]
        self.num_fragments = len(self.fragments)
        self.max_atoms = max_atoms  # Maximum number of atoms in the molecule
        self.action_space_size = 1 + (self.max_atoms * self.num_fragments)  # Include 'terminate' action
        self.max_steps = max_steps  # Maximum steps per episode
        self.curriculum_level = curriculum_level
        self.set_curriculum_level(curriculum_level)
        self.reset()
        self.Descriptors = Descriptors
        
        # Initialize additional attributes for enhanced rewards
        self.cumulative_performance = 0
        self.reward_stats = {
            # Initialize reward statistics for components
            # Example:
            'autophagy': {'mean': 0.0, 'std': 1.0},
            # Add other components as needed
        }
        self.diverse_mols = []

    def set_curriculum_level(self, level):
        self.curriculum_level = level
        if level == 1:
            self.max_steps = 10
            self.max_atoms = 20
            print("Curriculum level set to 1: Simple fragments")
        elif level == 2:
            self.max_steps = 15
            self.max_atoms = 25
            print("Curriculum level set to 2: Intermediate fragments")
        elif level == 2.5:
            self.max_steps = 12
            self.max_atoms = 23
            print("Curriculum level set to 2.5: Partial complex fragments")
        elif level == 3:
            self.max_steps = 20
            self.max_atoms = 30
            print("Curriculum level set to 3: Complex fragments, docking scores, pathway focus")
        else:
            print(f"Curriculum level {level} not recognized. No changes made.")

    def reset(self):
        # Start with a benzene ring as the starting molecule
        self.current_mol = Chem.RWMol(Chem.MolFromSmiles('C1=CC=CC=C1'))
        self.num_steps = 0
        self.current_step = 0
        self.invalid_molecule = False
        self.diverse_mols = []
        self.cumulative_performance = 0  # Reset cumulative performance
        print("Environment reset. Starting with a benzene ring.")
        return self.get_state()

    def get_state(self):
        mol = self.current_mol.GetMol()
        if mol.GetNumAtoms() > 0:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            state = np.array(fp)
        else:
            state = np.zeros(2048)
        return state

    def step(self, action):
        self.current_step += 1

        if action == 0:
            # Terminate the episode
            done = True
            reward = self._calculate_reward()
            return self.get_state(), reward, done, {}
        
        # Decode action to fragment addition
        fragment_idx = (action - 1) // self.max_atoms
        atom_position = (action - 1) % self.max_atoms

        if fragment_idx >= self.num_fragments:
            print(f"Invalid fragment index: {fragment_idx}")
            self.invalid_molecule = True
            done = True
            reward = -1.0  # Penalty for invalid action
            return self.get_state(), reward, done, {}

        fragment = self.fragments[fragment_idx]
        if fragment is None:
            print("Fragment is None!")
            self.invalid_molecule = True
            done = True
            reward = -1.0
            return self.get_state(), reward, done, {}

        # Attempt to add fragment to the molecule
        try:
            self.current_mol = Chem.RWMol(Chem.CombineMols(self.current_mol, fragment))
            self.num_steps += 1
            done = False
            reward = 0.0  # Placeholder for reward calculation
        except Exception as e:
            print(f"Error adding fragment: {e}")
            self.invalid_molecule = True
            done = True
            reward = -1.0  # Penalty for invalid action

        return self.get_state(), reward, done, {}

    def _calculate_reward(self):
        # Implement your reward calculation logic here
        # For example, based on QED score and other factors
        mol = self.current_mol.GetMol()
        qed_score = Descriptors.qed(mol)
        reward = qed_score  # Simple example: reward is QED score
        print(f"QED Score: {qed_score:.4f}")
        return reward

    def _extract_features(self, mol):
        descriptors = self._calculate_descriptors(mol)
        fingerprints = self._calculate_fingerprints(mol)
        return np.concatenate([descriptors, fingerprints])
    
    def _calculate_descriptors(self, mol):
        return np.array([
            Descriptors.ExactMolWt(mol),              # Molecular weight
            Descriptors.MolLogP(mol),                 # LogP
            Descriptors.NumHDonors(mol),              # Number of Hydrogen Bond Donors
            Descriptors.NumHAcceptors(mol),           # Number of Hydrogen Bond Acceptors
            Descriptors.TPSA(mol),                    # Topological Polar Surface Area
            Descriptors.NumRotatableBonds(mol),       # Number of Rotatable Bonds
            rdMolDescriptors.CalcNumRings(mol),       # Total number of rings
            rdMolDescriptors.CalcNumAromaticRings(mol),# Number of aromatic rings
            Descriptors.FractionCSP3(mol),            # Fraction of SP3 carbons
            mol.GetNumHeavyAtoms(),                   # Number of heavy atoms
            Descriptors.NumHeteroatoms(mol),          # Number of heteroatoms
            Descriptors.NumAliphaticRings(mol),       # Number of aliphatic rings
            Descriptors.NumAromaticCarbocycles(mol),  # Number of aromatic carbocycles
            Descriptors.NumAromaticHeterocycles(mol), # Number of aromatic heterocycles
            Descriptors.NumSaturatedRings(mol),       # Number of saturated rings
            Descriptors.NumAliphaticHeterocycles(mol),# Number of aliphatic heterocycles
            Descriptors.NumAliphaticCarbocycles(mol), # Number of aliphatic carbocycles
            Descriptors.NHOHCount(mol),               # Number of NHs or OHs
            Descriptors.NumRadicalElectrons(mol),     # Number of radical electrons
        ])

    def _calculate_fingerprints(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        array = np.zeros((1024,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

def update_curriculum_level(episode, episodes_per_level=1000, max_level=3):
    return min(max_level, (episode // episodes_per_level) + 1)

def select_action(policy, smiles):
    state = torch.tensor(smiles, dtype=torch.float).to(DEVICE)
    action_probs, state_value = policy(state)
    m = Categorical(action_probs)
    action = m.sample()
    policy.saved_actions.append((m.log_prob(action), state_value))
    return action.item()

def finish_episode(policy, optimizer, gamma=0.99, eps=1e-8):
    R = 0
    saved_actions = policy.saved_actions
    policy_losses = []
    value_losses = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(DEVICE)
    if returns.std() > eps:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    else:
        returns = returns - returns.mean()
    
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(DEVICE)))
    
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    
    optimizer.step()
    loss_value = loss.item()
    
    del policy.rewards[:]
    del policy.saved_actions[:]
    return loss_value

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f"Saved checkpoint: {filename}")

def load_checkpoint(filepath, model, optimizer):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at '{filepath}'")
    checkpoint = torch.load(filepath, map_location=DEVICE)
    try:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except Exception as e:
        raise RuntimeError(f"Error loading state_dict: {e}")
    start_episode = checkpoint.get('episode', 0)
    print(f"Loaded checkpoint '{filepath}' (Episode {start_episode})")
    return start_episode

def select_action(policy, state):
    state = torch.tensor(state, dtype=torch.float).to(DEVICE)
    action_probs, state_value = policy(state)
    m = Categorical(action_probs)
    action = m.sample()
    policy.saved_actions.append((m.log_prob(action), state_value))
    return action.item()



def main():
    parser = argparse.ArgumentParser(description='Molecular RL Agent')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=543, help='Random seed')
    parser.add_argument('--mode', choices=['train', 'inference'], default='train', help='Operation mode')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Initialize Policy and Optimizer
    policy = Policy(
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        hidden_size=HIDDEN_SIZE,
        action_size=ACTION_SIZE,
        dropout_p=DROPOUT_P,
        fine_tune=False
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=LEARNING_RATE
    )

    # Load Checkpoint if specified
    start_episode = 0
    if args.checkpoint is not None:
        try:
            start_episode = load_checkpoint(args.checkpoint, policy, optimizer)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    # Initialize Environment
    env = MoleculeEnv()

    # Generate timestamp and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"molecular_rl_agent_model_{timestamp}"
    log_file = f"{FILE_DIR}/{model_name}_training_log.csv"

    # Initialize Logging
    log_data = []
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as csvfile:
            fieldnames = ['Episode', 'Curriculum_Level', 'Reward', 'Running_Reward', 'Loss', 'QED_Score', 'SMILES']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    running_reward = 0
    for episode in range(start_episode + 1, args.episodes + 1):
        if args.mode == 'train':
            # Update curriculum level based on episode
            new_level = update_curriculum_level(episode)
            env.set_curriculum_level(new_level)

            # Reset environment
            state = env.reset()
            done = False
            ep_reward = 0

            while not done:
                smiles = Chem.MolToSmiles(env.current_mol.GetMol())
                action = select_action(policy, smiles)
                state, reward, done, _ = env.step(action)
                policy.rewards.append(reward)
                ep_reward += reward

            # Finish episode and update policy
            loss = finish_episode(policy, optimizer, gamma=0.99)

            # Update running reward
            running_reward = 0.15 * ep_reward + (1 - 0.15) * running_reward

            # Log progress
            qed_score = Descriptors.qed(env.current_mol.GetMol())
            final_smiles = Chem.MolToSmiles(env.current_mol.GetMol())
            log_entry = {
                'Episode': episode,
                'Curriculum_Level': env.curriculum_level,
                'Reward': ep_reward,
                'Running_Reward': running_reward,
                'Loss': loss,
                'QED_Score': qed_score,
                'SMILES': final_smiles
            }
            log_data.append(log_entry)

            # Write to CSV
            with open(log_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['Episode', 'Curriculum_Level', 'Reward', 'Running_Reward', 'Loss', 'QED_Score', 'SMILES'])
                writer.writerow(log_entry)

            print(f"Episode {episode} completed with reward {ep_reward:.4f}. Running Reward: {running_reward:.4f}")

            # Save Checkpoint
            if episode % SAVE_INTERVAL == 0:
                checkpoint_path = f"{FILE_DIR}/{model_name}_checkpoint_episode_{episode}.pth.tar"
                save_checkpoint({
                    'episode': episode,
                    'state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, filename=checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        elif args.mode == 'inference':
            # Inference mode
            policy.eval()  # Set to evaluation mode
            with torch.no_grad():
                state = env.reset()
                done = False
                ep_reward = 0
                while not done:
                    smiles = Chem.MolToSmiles(env.current_mol.GetMol())
                    action = select_action(policy, smiles)
                    state, reward, done, _ = env.step(action)
                    ep_reward += reward

                final_smiles = Chem.MolToSmiles(env.current_mol.GetMol())
                qed_score = Descriptors.qed(env.current_mol.GetMol())
                print(f"Generated molecule SMILES: {final_smiles}")
                print(f"QED Score: {qed_score:.4f}")
            break  # Exit after inference

    if args.mode == 'train':
        # Save Final Model
        final_checkpoint = f"{FILE_DIR}/{model_name}_final_model.pth.tar"
        save_checkpoint({
            'episode': args.episodes,
            'state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filename=final_checkpoint)
        print(f"Training completed. Final model saved to {final_checkpoint}")

        # Optionally log the entire training progress
        full_log_file = f"{FILE_DIR}/{model_name}_full_training_log.csv"
        log_training_progress(log_data, filename=full_log_file)

if __name__ == '__main__':
    main()