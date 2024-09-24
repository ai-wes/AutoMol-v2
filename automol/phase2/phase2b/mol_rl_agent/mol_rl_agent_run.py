
# main.py

# Import necessary modules
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import os
import sys
# molecule_env.py

import os
import sys



import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
from rdkit import DataStructs
import sascorer  # Ensure sascorer is installed
from docking_scorer import predict_docking_score  # Ensure this module is available
import csv
import os
from datetime import datetime

RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings
from transform_and_path import REACTIONS, PATHWAY_SCORING_FUNCTIONS, AGING_PATHWAYS
# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from rdkit.Chem import Descriptors
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import argparse
import csv
from transform_and_path import *
from policy import Policy
from docking_scorer import predict_docking_score
from generate_training_report import generate_training_report
from datetime import datetime

args = ""


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
"""
 _____              __ _                       _   _             
/  __ \            / _(_)                     | | (_)            
| /  \/ ___  _ __ | |_ _  __ _ _   _ _ __ __ _| |_ _  ___  _ __  
| |    / _ \| '_ \|  _| |/ _` | | | | '__/ _` | __| |/ _ \| '_ \ 
| \__/\ (_) | | | | | | | (_| | |_| | | | (_| | |_| | (_) | | | |
 \____/\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|\__|_|\___/|_| |_|
                          __/ |                                  
                         |___/                                   
"""
# config.py
import os
import torch

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Parameters
STATE_SIZE = 2048
HIDDEN_SIZE = 256
ACTION_SIZE = 1 + (20 * 21)  # Termination action + (max_atoms * num_fragments)
DROPOUT_P = 0.2
PRETRAINED_MODEL_NAME = 'mrm8488/chEMBL_smiles_v1'

# Training Parameters
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
SAVE_INTERVAL = 1000
# At the top of the file, add:
from datetime import datetime

# In the configuration section, update:os.makedirs(SAVE_DIR, exist_ok=True)
FRAGMENTS_BY_CURRICULUM = {
    1: [
        'C',  # methyl group
        'O',  # oxygen atom
        'N',  # nitrogen atom
        'S',  # sulfur atom
        'Cl',  # chlorine atom
        'F',  # fluorine atom
        'Br',  # bromine atom
        'I',  # iodine atom
    ],
    2: [
        'C=O',  # carbonyl group
        'C#N',  # nitrile group
        'C=C',  # alkene group
        'C#C',  # alkyne group
        'C1CC1',  # cyclopropane ring
        'C1CCC1',  # cyclobutane ring
        'C1CCCC1',  # cyclopentane ring
        'C1CCCCC1',  # cyclohexane ring
        'c1ccccc1',  # benzene ring
    ],
    2.5: [
        'C(=O)O',  # carboxylic acid group
        'C(=O)N',  # amide group
        'C1=CC=CC=C1',  # another benzene ring
        'C1=CC=CN=C1',  # pyridine ring
        'C1=CC=CC=N1',  # pyridine ring (alternative notation)
        'C1=CC=CC=C1O',  # phenol
        'C1=CC=CC=C1N',  # aniline
        'C1=CC=CC=C1C(=O)O',  # benzoic acid
    ],
    3: [
        'C1=CC=C(C=C1)C(=O)O',  # benzoic acid
        'C1=CC=C(C=C1)C(=O)N',  # benzamide
        'C1=CC=C(C=C1)C(=O)C',  # acetophenone
        'C1=CC=C(C=C1)C(=O)CC',  # propiophenone
        'C1=CC=C(C=C1)C(=O)CCC',  # butyrophenone
        'C1=CC=C(C=C1)C(=O)CCCC',  # valerophenone
        'C1=CC=C(C=C1)C(=O)CCCCC',  # hexanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCC',  # heptanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCC',  # octanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCC',  # nonanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCC',  # decanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCC',  # undecanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCCC',  # dodecanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCC',  # tridecanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCCC',  # tetradecanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCCCC',  # pentadecanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCCCCC',  # hexadecanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCCCCCC',  # heptadecanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCCCCCCC',  # octadecanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCCCCCCCC',  # nonadecanophenone
        'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCCCCCCCCC',  # eicosanophenone
    ]
}


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

"""
 __  __       _                 _        ______            
|  \/  |     | |               | |      |  ____|           
| \  / | ___ | | ___  ___ _   _| | ___  | |__   _ ____   __
| |\/| |/ _ \| |/ _ \/ __| | | | |/ _ \ |  __| | '_ \ \ / /
| |  | | (_) | |  __/ (__| |_| | |  __/ | |____| | | \ V / 
|_|  |_|\___/|_|\___|\___|\__,_|_|\___| |______|_| |_|\_/  


"""
class MoleculeEnv:
    def __init__(self, max_steps=10, max_atoms=20, curriculum_level=1, reactions=REACTIONS, pathway_scoring_functions=PATHWAY_SCORING_FUNCTIONS):
        """
        Initializes the Molecule Environment for RL agent.
        
        Args:
            max_steps (int): Maximum steps per episode.
            max_atoms (int): Maximum number of atoms in a molecule.
            curriculum_level (int): Current curriculum level.
            reactions (list): List of reaction tuples.
            pathway_scoring_functions (dict): Dictionary of pathway scoring functions.
        """
        self.fragments = [Chem.MolFromSmiles(frag) for frag in fragments]
        print(f"Fragments initialized: {len(self.fragments)} fragments")
        
        self.num_fragments = len(self.fragments)
        print(f"Number of fragments: {self.num_fragments}")
        
        self.max_atoms = max_atoms  # Maximum number of atoms in the molecule
        print(f"Maximum atoms set to: {self.max_atoms}")
        
        self.action_space_size = 1 + (self.max_atoms * self.num_fragments)  # Include 'terminate' action
        print(f"Action space size: {self.action_space_size}")
        
        self.max_steps = max_steps  # Maximum steps per episode
        print(f"Maximum steps per episode: {self.max_steps}")
        
        self.curriculum_level = curriculum_level
        print(f"Initial curriculum level: {self.curriculum_level}")
        
        self.set_curriculum_level(curriculum_level)
        print(f"Curriculum level set to: {self.curriculum_level}")
        
        # Initialize pathway attributes
        self.current_pathway_index = 0  # Initialize pathway index
        print(f"Current pathway index initialized to: {self.current_pathway_index}")
        
        self.pathways = list(pathway_scoring_functions.keys())  # List of pathways
        print(f"Pathways initialized: {len(self.pathways)} pathways")
        
        self.current_pathway = self.pathways[self.current_pathway_index]  # Set initial pathway
        print(f"Current pathway set to: {self.current_pathway}")
        
        # Combine initial SMILES with pathway SMILES
        self.initial_smiles = [
            'C1=CC=CC=C1',        # Benzene
            'CCO',                # Ethanol
            'CCN',                # Ethylamine
            'CCC',                # Propane
            'CC(=O)O',            # Acetic acid
            'CC(=O)N',            # Acetamide
            'CC#N',               # Acetonitrile
            'C1CCCCC1',           # Cyclohexane
            'C1=CC=CC=C1O',       # Phenol
            'CC(C)O',             # Isopropanol
            # Add more diverse SMILES as needed
        ] 
        
        
        
        """[
            "CC1=CC=C(C=C1)NC(=O)C",
            "COC1=C(C=C(C=C1)NC(=O)C)OC",
            "CC(=O)NC1=CC=C(C=C1)O",
            "CC1=C(C(=CC=C1)C)NC(=O)CC2=CC=CC=C2",
            "COC1=CC(=CC(=C1OC)OC)C(=O)NC2=CC=CC=C2",
            "CC1=CC=C(C=C1)NC(=O)CCCN2CCC(CC2)N3C(=O)NC4=CC=CC=C43",
            "COC1=C(C=C(C=C1)NC(=O)C2CC3=C(NC2=O)C=CC(=C3)Cl)OCCCN4CCOCC4",
            "CC(=O)NC1=CC=CC=C1",
            "CC1=C(C=C(C=C1)S(=O)(=O)NC2=CC=CC=C2)C",
            "CC1=CC(=NO1)C2=CC(=CC=C2)S(=O)(=O)NC3=CC=CC=C3",
            "CC1=C(C(=O)NC(=O)N1)C2=CC=CC=C2",
            "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C3=CC=CC=C3",
            "CC1=C(NC(=O)C2=C1C=CC(=C2)C#N)C3=CC=C(C=C3)NC(=O)C4=CC=CC=C4",
            "C1=CC(=CC=C1C(=O)O)O",
            "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=O)O)C",
            "CC1=C(C(=O)C2=C(C1=O)C(=CC=C2)O)O",
            "CC1=CC=C(C=C1)OCC(C)(C)NC2=NC=NC3=C2C=CN3",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCOCC3)OC",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "CC(=O)Nc1ccc(cc1)C(=O)O",
            "C1=CC=CC=C1C(=O)O",
            "CCN(CC)C(=O)C1=CC=CC=C1",
            "CC(=O)Nc1ccc(cc1)N",
            "CCOCC(=O)Nc1ccc(cc1)O",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "CC1=C(C(=O)N2CCCC2=N1)C3=CC=C(C=C3)Cl",
            "CC1=NC(=C(N1CC2=CC=C(C=C2)C(=O)O)C(=O)NC3=CC=CC=C3)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
            "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "C1=NC2=C(N1)C(=O)NC(=O)N2",
            "C1=NC(=O)NC(=O)C1",
            "CCC(=O)Nc1ccccc1O",
            "CC(=O)NCC1=CC=CC=C1",
            "CC(=O)Nc1ccc(cc1)O",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "C1=NC(=C2C(=N1)C(=O)NC(=O)N2)N",
            "C1=NC2=C(N1)C(=O)NC(=O)N2",
            "CC(=O)NC1=CC=C(C=C1)O",
            "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
            "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
            "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC"
        ]"""

        self.current_episode = 0  # To keep track of which SMILES to use
        print(f"Initialized current_episode: {self.current_episode}")
        
        self.difficulty = 1  # Initialize difficulty attribute
        print(f"Initialized difficulty: {self.difficulty}")

        self.reward_stats = {
            'docking': {'mean': 0.0, 'std': 1.0},
            'pathway': {'mean': 0.0, 'std': 1.0},
            'multi_target': {'mean': 0.0, 'std': 1.0},
            'qed': {'mean': 0.0, 'std': 1.0},
            'weight': {'mean': 0.0, 'std': 1.0},
            'logp': {'mean': 0.0, 'std': 1.0},
            'sas': {'mean': 0.0, 'std': 1.0},
            'lipinski': {'mean': 0.0, 'std': 1.0},
            'diversity': {'mean': 0.0, 'std': 1.0},
            'step': {'mean': 0.0, 'std': 1.0},
        }
        print(f"Initialized reward_stats: {self.reward_stats}")
        self.smiles_sequence = []

        self.reset()  # Now call reset after initializing initial_smiles
        print("Called reset() method")

    def reset(self):
        """
        Resets the environment for a new episode.
        
        Returns:
            mol (rdkit.Chem.Mol): The starting molecule.
        """
        # Select the starting SMILES in a cyclical manner
        starting_smiles = self.initial_smiles[self.current_episode % len(self.initial_smiles)]
        print(f"Selected starting SMILES: {starting_smiles}")
        
        self.current_mol = Chem.RWMol(Chem.MolFromSmiles(starting_smiles))
        print(f"Initialized current_mol with SMILES: {Chem.MolToSmiles(self.current_mol)}")
        
        self.num_steps = 0
        print(f"Reset num_steps to: {self.num_steps}")
        
        self.current_step = 0
        print(f"Reset current_step to: {self.current_step}")
        
        self.invalid_molecule = False
        print(f"Reset invalid_molecule to: {self.invalid_molecule}")
        
        self.diverse_mols = []
        print(f"Reset diverse_mols to: {self.diverse_mols}")
        
        self.cumulative_performance = 0  # Reset cumulative performance
        print(f"Reset cumulative_performance to: {self.cumulative_performance}")
        
        # Update for next episode
        self.current_episode += 1
        print(f"Incremented current_episode to: {self.current_episode}")
        
        # Optionally, update the current pathway based on curriculum level
        self.current_pathway = self.pathways[self.current_pathway_index]
        print(f"Updated current_pathway to: {self.current_pathway}")
        
        self.smiles_sequence = [Chem.MolToSmiles(self.current_mol)]
        return self.current_mol.GetMol()

    # ... rest of the MoleculeEnv methods ...

    # ... rest of the MoleculeEnv methods ...

                
    """
    _____  _                                 _   
    |  __ \| |          /\                   | |  
    | |__) | |         /  \   __ _  ___ _ __ | |_ 
    |  _  /| |        / /\ \ / _` |/ _ \ '_ \| __|
    | | \ \| |____   / ____ \ (_| |  __/ | | | |_ 
    |_|  \_\______| /_/    \_\__, |\___|_| |_|\__|
                            __/ |               
                            |___/                
    """



    def step(self, action):
        done = False
        reward = 0.0  # Initialize reward

        if action == 0:
            # Terminate episode
            reward = self._compute_reward(mol=self.current_mol.GetMol())
            done = True
            state = self.get_state()
            print(f"Episode terminated. Final reward: {reward}")
            return state, reward, done, {}

        # Process the action
        action_index = action - 1  # Shift action to account for termination action
        atom_index = action_index // self.num_fragments
        fragment_index = action_index % self.num_fragments

        # Check if the atom_index is valid
        if atom_index >= self.current_mol.GetNumAtoms():
            # Invalid action
            reward = -0.2  # Penalty for invalid action
            print(f"Invalid action: Atom index {atom_index} is out of range. Penalty reward: {reward}")
            return self.get_state(), reward, done, {}

        # Get the fragment to add
        fragment = self.fragments[fragment_index]
        new_mol = Chem.RWMol(self.current_mol)

        # Add atoms from fragment
        frag_atom_mapping = {}
        for atom in fragment.GetAtoms():
            new_atom_idx = new_mol.AddAtom(atom)
            frag_atom_mapping[atom.GetIdx()] = new_atom_idx

        # Find the attachment point (assume the first atom in the fragment)
        frag_attachment_idx = frag_atom_mapping[0]

        # Try to add a bond between the specified atom in the molecule and the fragment
        try:
            new_mol.AddBond(atom_index, frag_attachment_idx, Chem.rdchem.BondType.SINGLE)
            Chem.SanitizeMol(new_mol)
            self.current_mol = new_mol  # Update the current molecule
            print(f"Successfully added fragment to atom {atom_index}")
        except Exception as e:
            # Invalid action
            reward = -0.25  # Penalty for invalid action
            print(f"Invalid action. Failed to add fragment. Penalty reward: {reward}. Error: {e}")
            return self.get_state(), reward, done, {}

        # Increment steps and check for termination
        self.num_steps += 1
        self.current_step += 1

        if self.num_steps >= self.max_steps or self.current_mol.GetNumAtoms() >= self.max_atoms:
            done = True
            reward = self._compute_reward(mol=self.current_mol.GetMol())
            print(f"\n\n*********Episode ended. Steps: {self.num_steps}, Atoms: {self.current_mol.GetNumAtoms()}, Reward: {reward}************\n\n  ")
        else:
            reward = 0.0  # No immediate reward for successful step
            print(f"Step {self.num_steps} completed. Current molecule has {self.current_mol.GetNumAtoms()} atoms.")

        state = self.get_state()
        self.smiles_sequence.append(Chem.MolToSmiles(self.current_mol))
        return self.current_mol.GetMol(), reward, done, {}


    def _compute_reward(self, mol):
        smiles = Chem.MolToSmiles(mol)
        reward_components = {}
        # Docking score
        if self.curriculum_level >= 3:
            try:
                docking_score = predict_docking_score(smiles)
                reward_components['docking'] = -docking_score / 10  # Normalize docking reward (lower is better)
            except Exception as e:
                print(f"Error in predicting docking score: {e}")
                reward_components['docking'] = 0.0
        else:
            reward_components['docking'] = 0.0

        # Pathway targeting
        pathway_scores = self._get_pathway_scores(mol)
        reward_components['pathway'] = np.mean(pathway_scores) * 10

        # Multi-target bonus
        pathway_scores_array = np.array(pathway_scores)
        active_pathways = np.sum(pathway_scores_array > 0.9)  # Number of pathways hit
        reward_components['multi_target'] = (active_pathways / len(AGING_PATHWAYS)) * 2.5

        # Other molecular properties
        reward_components.update({
            'qed': Descriptors.qed(mol) * 2,
            'weight': -abs(Descriptors.MolWt(mol) - 400) / 200,
            'logp': -abs(Descriptors.MolLogP(mol) - 2.5) / 2.5,
            'sas': -sascorer.calculateScore(mol) / 10,
            'lipinski': -self._lipinski_violations(mol) * 0.5,
            'diversity': self._calculate_diversity() * 2,
            'step': max(0.5 - (self.current_step / 10), 0.2),
        })

        # Apply difficulty multiplier
        for key in reward_components:
            reward_components[key] *= (2 + self.difficulty)

        # Normalize rewards
        normalized_components = {}
        for component, value in reward_components.items():
            if component == 'pathway':
                normalized_value = value  # Already normalized
            else:
                self._update_reward_stats(component, value)
                stats = self.reward_stats[component]
                normalized_value = (value - stats['mean']) / (stats['std'] + 1e-8)
            normalized_components[component] = normalized_value

        # Calculate total reward
        total_reward = sum(normalized_components.values())
        total_reward = max(total_reward, -0.1)  # Limit minimum reward

        # Update cumulative performance for pathway scores
        self.cumulative_performance += np.array(pathway_scores)

        # Logging detailed reward components
        print(f"Reward components: " + ", ".join([f"{k}={v:.4f}" for k, v in reward_components.items()]))
        print(f"Normalized reward components: " + ", ".join([f"{k}={v:.4f}" for k, v in normalized_components.items()]))
        print(f"\n\n*********Total normalized reward: {total_reward:.4f}************\n\n")
        total_reward *= self.difficulty

        return total_reward

    def _get_pathway_scores(self, mol):
        scores = []
        for pathway in AGING_PATHWAYS:
            scoring_function = PATHWAY_SCORING_FUNCTIONS[pathway]
            score = scoring_function(mol)
            scores.append(score if score is not None else 0)
        return scores

########################################################################################################################################################
    


    def _lipinski_violations(self, mol):
        violations = 0
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
        return violations



    def _update_reward_stats(self, component, value):
        if 'mean' not in self.reward_stats[component] or 'std' not in self.reward_stats[component]:
            self.reward_stats[component] = {'mean': value, 'std': 1.0}
        else:
            # Update mean and std using exponential moving average
            self.reward_stats[component]['mean'] = 0.99 * self.reward_stats[component]['mean'] + 0.01 * value
            self.reward_stats[component]['std'] = 0.99 * self.reward_stats[component]['std'] + 0.01 * abs(value - self.reward_stats[component]['mean'])



    def _calculate_diversity(self):
        mol = self.current_mol.GetMol()
        current_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        similarities = []
        for existing_mol in self.diverse_mols:
            existing_fp = AllChem.GetMorganFingerprintAsBitVect(existing_mol, 2, nBits=1024)
            similarity = DataStructs.TanimotoSimilarity(current_fp, existing_fp)
            similarities.append(similarity)
        if similarities:
            diversity = 1 - max(similarities)
        else:
            diversity = 1.0  # Maximum diversity if there are no other molecules to compare
        return diversity
    
    
    
    
########################################################################################################################################################
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



########################################################################################################################################################


        
    def get_state(self):
        mol = self.current_mol.GetMol()
        if mol.GetNumAtoms() > 0:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_array = np.array(fp)
        else:
            fp_array = np.zeros(2048)
        
        # One-hot encode the current pathway
        pathway_index = self.pathways.index(self.current_pathway)
        pathway_one_hot = np.zeros(len(self.pathways))
        pathway_one_hot[pathway_index] = 1
        
        # Concatenate fingerprint and pathway one-hot encoding
        state = np.concatenate([fp_array, pathway_one_hot])
        return state
        
        
        

    def set_curriculum_level(self, level):
        self.curriculum_level = level
        if level in FRAGMENTS_BY_CURRICULUM:
            self.fragments = [Chem.MolFromSmiles(frag) for frag in FRAGMENTS_BY_CURRICULUM[level]]
            self.num_fragments = len(self.fragments)
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
                
########################################################################################################################################################

def update_curriculum_level(episode, episodes_per_level, curriculum_levels):
    max_level = len(curriculum_levels)
    return min(max_level, (episode // episodes_per_level) + 1)

########################################################################################################################################################

"""
 _    _ _   _ _     
| |  | | | (_) |    
| |  | | |_ _| |___ 
| |  | | __| | / __|
| |__| | |_| | \__ \
 \____/ \__|_|_|___/

"""

def select_action(policy, mol):
    if not isinstance(mol, Chem.Mol):
        raise TypeError(f"Expected rdkit.Chem.Mol object, got {type(mol)}")
    smiles = Chem.MolToSmiles(mol)
    action_probs, state_value = policy([smiles])  # Pass as a list of SMILES strings
    m = Categorical(action_probs)
    action = m.sample()
    policy.saved_actions.append((m.log_prob(action), state_value))
    return action.item()


def finish_episode(policy, optimizer, gamma=0.99, eps=1e-8):
    """
    Finish the episode by performing backpropagation and updating the policy network.
    
    Args:
        policy (Policy): The policy network.
        optimizer (torch.optim.Optimizer): The optimizer.
        gamma (float): Discount factor.
    
    Returns:
        loss (float): The loss value.
    """
    R = 0
    policy_losses = []
    value_losses = []
    returns = []

    # Calculate the discounted returns
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(policy.saved_actions, returns):
        advantage = R - value.item()

        # Calculate policy loss
        policy_losses.append(-log_prob * advantage)

        # Calculate value loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))

    # Ensure policy_losses and value_losses are not empty
    if policy_losses and value_losses:
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clear the action and reward buffers
        del policy.rewards[:]
        del policy.saved_actions[:]

        return loss.item()
    else:
        print("Warning: No actions or rewards collected during the episode.")
        return 0.0


def log_training_progress(log_data, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_log_{timestamp}.csv"
    
    fieldnames = ['Episode', 'Curriculum_Level', 'Reward', 'Running_Reward', 'Loss', 'QED_Score', 'SMILES_Sequence']
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in log_data:
            writer.writerow(entry)
    
    print(f"Training log saved to {filename}")



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Saved checkpoint: {filename}")


def load_checkpoint(filepath, model, optimizer, load_optimizer=True):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at '{filepath}'")
    checkpoint = torch.load(filepath, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
        if load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except Exception as e:
        raise RuntimeError(f"Error loading state_dict: {e}")
    start_episode = checkpoint.get('episode', 0)
    running_reward = checkpoint.get('running_reward', 0)
    curriculum_level = checkpoint.get('curriculum_level', 1)
    print(f"Loaded checkpoint '{filepath}'")
    print(f"Checkpoint was saved at Episode {start_episode}")
    print(f"Checkpoint Running Reward: {running_reward}")
    print(f"Checkpoint Curriculum Level: {curriculum_level}")
    return start_episode, running_reward, curriculum_level

import json

import json
from termcolor import colored

def save_best_run(model_name, avg_reward, final_checkpoint, log_file, txt_report, html_report):
    best_run_file = 'best_run.json'
    best_run_info = {
        'model_name': model_name,
        'average_reward': avg_reward,
        'model_checkpoint': final_checkpoint,
        'log_file': log_file,
        'txt_report': txt_report,
        'html_report': html_report
    }
    
    try:
        with open(best_run_file, 'r') as f:
            current_best = json.load(f)
        current_best_reward = current_best['average_reward']
    except FileNotFoundError:
        current_best_reward = float('-inf')

    if avg_reward > current_best_reward:
        with open(best_run_file, 'w') as f:
            json.dump(best_run_info, f, indent=4)
        print(colored(f"New best run! Average reward: {avg_reward:.4f}", "green"))
        print(colored(f"Improvement: +{avg_reward - current_best_reward:.4f}", "green"))
    else:
        print(colored(f"This run ({avg_reward:.4f}) did not beat the current best ({current_best_reward:.4f})", "red"))
        print(colored(f"Difference: {avg_reward - current_best_reward:.4f}", "red"))

    # Always print the current run's average reward for reference
    print(f"Current run average reward: {avg_reward:.4f}")
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
"""
 __  __       _       
|  \/  |     (_)      
| \  / | __ _ _ _ __  
| |\/| |/ _` | | '_ \ 
| |  | | (_| | | | | |
|_|  |_|\__,_|_|_| |_|

"""
def main():
    parser = argparse.ArgumentParser(description='Molecular RL Agent Training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--load_optimizer', action='store_true', help='Load optimizer state from checkpoint')
    parser.add_argument('--continue_from_checkpoint', action='store_true', help='Continue training from checkpoint episode')
    parser.add_argument('--total_episodes', type=int, default=5000, help='Total number of training episodes')
    parser.add_argument('--episodes_per_curriculum', type=int, default=1000, help='Number of episodes per curriculum level')
    parser.add_argument('--curriculum_levels', type=str, default='1,2,2.5,3', help='Comma-separated list of curriculum levels to use')
    parser.add_argument('--seed', type=int, default=543, help='Random seed')
    args = parser.parse_args()

    # Parse curriculum levels
    curriculum_levels = [float(level) for level in args.curriculum_levels.split(',')]
    
    # Create a unique output directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"molecular_rl_agent_model_{timestamp}"
    output_dir = os.path.join("logs", "training_reports", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths within the output directory
    log_file = os.path.join(output_dir, f"{model_name}_training_log.csv")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    report_dir = os.path.join(output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    # Initialize Environment with the first curriculum level
    env = MoleculeEnv(curriculum_level=curriculum_levels[0], 
                      reactions=REACTIONS, 
                      pathway_scoring_functions=PATHWAY_SCORING_FUNCTIONS)

    # Initialize Policy and Optimizer
    policy = Policy(
        pretrained_model_name='mrm8488/chEMBL_smiles_v1',
        hidden_size=256,
        action_size=421,
        dropout_p=0.2,
        fine_tune=False
    ).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=1e-4
    )

    # Load Checkpoint if specified
    if args.checkpoint is not None:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            policy.load_state_dict(checkpoint['state_dict'])
            if args.load_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint '{args.checkpoint}'")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    log_data = []

    # Initialize Logging
    fieldnames = ['Episode', 'Curriculum_Level', 'Reward', 'Running_Reward', 'Loss', 'QED_Score', 'SMILES_Sequence', 'Pathway', 'Pathway_Score', 'Diversity_Score', 'Multitarget_Score']
    
    # Write CSV header
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Training Loop
    start_episode = 0
    total_reward = 0
    running_reward = 0
    log_data = []  # Initialize log_data list
    for episode in range(start_episode + 1, args.total_episodes + 1):
        # Update curriculum level based on episode
        curriculum_index = update_curriculum_level(episode, args.episodes_per_curriculum, curriculum_levels)
        if curriculum_index < len(curriculum_levels):
            env.set_curriculum_level(curriculum_levels[curriculum_index])

        state_mol = env.reset()  # rdkit.Chem.Mol object
        done = False
        ep_reward = 0
        step = 0
        
        while not done:
            try:
                action = select_action(policy, state_mol)  # Pass the molecule object
            except ValueError as ve:
                print(f"Error selecting action: {ve}")
                reward = -1.0  # Penalize invalid state
                done = True
                break

            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            # Update state_mol to maintain it as a rdkit.Chem.Mol object
            state_mol = env.current_mol.GetMol()
            step += 1

        # Finish episode and update policy
        loss = finish_episode(policy, optimizer, gamma=0.99)

        # Update running reward
        old_running_reward = running_reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # Generate log entry
        log_entry = {
            'Episode': episode,
            'Curriculum_Level': env.curriculum_level,
            'Reward': ep_reward,
            'Running_Reward': running_reward,
            'Loss': loss,
            'QED_Score': Descriptors.qed(state_mol),
            'SMILES_Sequence': '|'.join(env.smiles_sequence),
            'Pathway': env.current_pathway,
            'Pathway_Score': np.mean(env._get_pathway_scores(state_mol)),
            'Diversity_Score': env._calculate_diversity(),
            'Multitarget_Score': sum(score > 0.9 for score in env._get_pathway_scores(state_mol)) / len(AGING_PATHWAYS)
        }

        total_reward += ep_reward
        log_data.append(log_entry)

        # Append to CSV file every 50 episodes
        if episode % 50 == 0:
            with open(log_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(log_data)
                log_data = []  # Clear log data after writing

        # Print progress every 100 episodes
        if episode % 100 == 0 or episode == start_episode + 1:
            reward_diff = running_reward - old_running_reward
            reward_color = "green" if reward_diff >= 0 else "red"
            print(f"Episode {episode} completed with reward {ep_reward:.4f}.")
            print(colored(f"Running Reward: {running_reward:.4f} ({reward_diff:+.4f})", reward_color))
            print(f"Curriculum Level: {env.curriculum_level}, Steps: {step}, SMILES: {log_entry['SMILES_Sequence']}")
                
        # Save Checkpoint
        if episode % 5000 == 0:
            checkpoint_path = os.path.join(model_dir, f"{model_name}_checkpoint_episode_{episode}.pth.tar")
            save_checkpoint({
                'episode': episode,
                'state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'running_reward': running_reward,
                'curriculum_level': env.curriculum_level,
            }, filename=checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save Final Model
    final_checkpoint = os.path.join(model_dir, f"{model_name}_final_model.pth.tar")
    save_checkpoint({
        'episode': args.total_episodes,
        'state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename=final_checkpoint)

    # Generate training report
    report_dir = os.path.join(report_dir)  # Reports are already in output_dir/reports
    os.makedirs(report_dir, exist_ok=True)
    txt_report, html_report = generate_training_report(log_file, report_dir, best_run_path='best_run.json')

    # Calculate average reward for the entire run
    avg_reward = total_reward / args.total_episodes

    print(f"Total reward: {total_reward:.4f}")
    print(f"Average reward: {avg_reward:.4f}")

    # Save Best Run
    save_best_run(model_name, avg_reward, final_checkpoint, log_file, txt_report, html_report)
    
    # Optionally, save the raw training log CSV in the output_dir (already saved)

if __name__ == '__main__':
    main()