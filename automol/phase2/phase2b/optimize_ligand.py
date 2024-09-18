import logging
import json
import os
from typing import Dict, Any, List, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data
import torch.nn.functional as F

# Import other necessary modules
import numpy as np
import random

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch

class EnhancedTelomeraseGNN(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        num_heads: int,
        num_node_types: int
    ):
        super(EnhancedTelomeraseGNN, self).__init__()
        self.node_embedding = nn.Embedding(num_node_types, hidden_channels)
        self.edge_embedding = nn.Linear(num_edge_features, hidden_channels)
        self.conv1 = GATConv(
            in_channels=num_node_features + hidden_channels, 
            out_channels=hidden_channels, 
            heads=num_heads, 
            edge_dim=hidden_channels, 
            concat=False
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels, 
            out_channels=hidden_channels, 
            heads=num_heads, 
            edge_dim=hidden_channels, 
            concat=False
        )
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.telomerase_activity_head = nn.Linear(hidden_channels, 1)
        self.compound_effectiveness_head = nn.Linear(hidden_channels, 1)
        self.pchembl_value_head = nn.Linear(hidden_channels, 1)

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor, 
        batch: torch.Tensor, 
        node_type: torch.Tensor
    ):
        node_embed = self.node_embedding(node_type)
        x = torch.cat([x, node_embed], dim=-1)

        edge_embed = self.edge_embedding(edge_attr)

        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_embed))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_embed))
        x = F.elu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        telomerase_activity = self.telomerase_activity_head(x)
        compound_effectiveness = self.compound_effectiveness_head(x)
        pchembl_value = self.pchembl_value_head(x)

        return telomerase_activity, compound_effectiveness, pchembl_value



def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> EnhancedTelomeraseGNN:

    try:
        logger.debug(f"Loading model from checkpoint: {checkpoint_path}")
        model = EnhancedTelomeraseGNN(
            num_node_features=config['num_node_features'],
            num_edge_features=config['num_edge_features'],
            hidden_channels=config['hidden_channels'],
            num_heads=config['num_heads'],
            num_node_types=config['num_node_types']
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        logger.info("Model loaded successfully.")
        print("Model loaded successfully.")
        return model
    except FileNotFoundError as fnf_error:
        logger.error(f"Checkpoint file not found: {fnf_error}")
        print(f"Checkpoint file not found: {fnf_error}")
    except KeyError as key_error:
        logger.error(f"Missing configuration key: {key_error}")
        print(f"Missing configuration key: {key_error}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print(f"Failed to load model: {e}")
    return None

def protein_to_graph_data(protein_sequence: str) -> Data:

    try:
        logger.debug("Converting protein sequence to graph data.")
        print("Converting protein sequence to graph data.")
        # Placeholder implementation: Create dummy graph data
        num_nodes = len(protein_sequence)
        x = torch.arange(num_nodes).unsqueeze(1).float()  # Node features
        edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)]).t().contiguous()
        edge_attr = torch.ones(edge_index.size(1), 1)  # Edge features
        batch = torch.zeros(num_nodes, dtype=torch.long)  # Single graph
        node_type = torch.zeros(num_nodes, dtype=torch.long)  # Dummy node types
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, node_type=node_type)
        logger.debug("Protein sequence converted to graph data successfully.")
        print("Protein sequence converted to graph data successfully.")
        return data
    except Exception as e:
        logger.error(f"Error converting protein sequence to graph data: {e}")
        print(f"Error converting protein sequence to graph data: {e}")
        # Return an empty Data object in case of failure
        return Data()


def apply_action(smiles: str, action: Tuple[str, int, str]) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.info(f"Invalid SMILES encountered: {smiles}")
        return None

    action_type, atom_idx, value = action
    
    try:
        atom = mol.GetAtomWithIdx(atom_idx)
        if action_type == 'add_hydrogen':
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
        elif action_type == 'remove_hydrogen':
            if atom.GetNumExplicitHs() > 0:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
        elif action_type == 'change_bond':
            for bond in atom.GetBonds():
                if bond.GetEndAtomIdx() == atom_idx or bond.GetBeginAtomIdx() == atom_idx:
                    if value == 'SINGLE':
                        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                    elif value == 'DOUBLE':
                        bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
                    break  # Change only one bond

        # Update the molecule to reflect the changes
        Chem.SanitizeMol(mol)
        new_smiles = Chem.MolToSmiles(mol)
        logging.debug(f"Action {action} applied successfully. New SMILES: {new_smiles}")
        return new_smiles
    except Exception as e:
        logging.info(f"Unable to apply action {action} to SMILES {smiles}: {str(e)}")
        return None
    



def molecule_to_graph_data(mol: Chem.Mol) -> Data:

    try:
        logger.debug("Converting molecule to graph data.")
        print("Converting molecule to graph data.")
        # Placeholder implementation using RDKit descriptors
        atoms = mol.GetAtoms()
        num_atoms = len(atoms)
        x = torch.tensor([atom.GetAtomicNum() for atom in atoms], dtype=torch.long).unsqueeze(1)
        bonds = mol.GetBonds()
        edge_index = []
        edge_attr = []
        for bond in bonds:
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            edge_attr.append([bond.GetBondTypeAsDouble()])
            edge_attr.append([bond.GetBondTypeAsDouble()])
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr = torch.empty((0,1), dtype=torch.float)
        batch = torch.zeros(num_atoms, dtype=torch.long)
        node_type = torch.zeros(num_atoms, dtype=torch.long)  # Dummy node types
        data = Data(x=x.float(), edge_index=edge_index, edge_attr=edge_attr, batch=batch, node_type=node_type)
        logger.debug("Molecule converted to graph data successfully.")
        print("Molecule converted to graph data successfully.")
        return data
    except Exception as e:
        logger.error(f"Error converting molecule to graph data: {e}")
        print(f"Error converting molecule to graph data: {e}")
        # Return an empty Data object in case of failure
        return Data()

class MCTSNode:

    def __init__(self, state: str, protein_sequence: str, parent=None):
        self.state = state
        self.protein_sequence = protein_sequence
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        mol = Chem.MolFromSmiles(state)
        if mol is None:
            logger.warning(f"Invalid SMILES: {state}")
            print(f"Invalid SMILES encountered: {state}")
            self.untried_actions = []
        else:
            self.untried_actions = get_valid_modifications(mol, protein_sequence)

    def is_fully_expanded(self) -> bool:

        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.414) -> 'MCTSNode':

        try:
            choices_weights = [
                (child.value / (child.visits + 1e-5)) +
                c_param * np.sqrt((2 * np.log(self.visits + 1)) / (child.visits + 1e-5))
                for child in self.children
            ]
            best = self.children[np.argmax(choices_weights)]
            logger.debug(f"Best child selected with SMILES: {best.state}")
            print(f"Best child selected with SMILES: {best.state}")
            return best
        except Exception as e:
            logger.error(f"Error selecting best child: {e}")
            print(f"Error selecting best child: {e}")
            return self

    def rollout(self, model: EnhancedTelomeraseGNN, protein_data: Data) -> float:

        try:
            current_state = self.state
            logger.debug(f"Starting rollout from state: {current_state}")
            print(f"Starting rollout from state: {current_state}")
            for i in range(5):
                mol = Chem.MolFromSmiles(current_state)
                if not mol:
                    logger.debug(f"Rollout stopped at iteration {i+1}: Invalid molecule.")
                    print(f"Rollout stopped at iteration {i+1}: Invalid molecule.")
                    break
                if Descriptors.MolWt(mol) > 500:
                    logger.debug(f"Rollout stopped at iteration {i+1}: Molecular weight > 500.")
                    print(f"Rollout stopped at iteration {i+1}: Molecular weight > 500.")
                    break
                possible_moves = get_valid_modifications(mol, self.protein_sequence)
                if not possible_moves:
                    logger.debug(f"Rollout stopped at iteration {i+1}: No valid modifications.")
                    print(f"Rollout stopped at iteration {i+1}: No valid modifications.")
                    break
                action = random.choice(possible_moves)
                new_state = apply_action(current_state, action)
                if new_state is None:
                    logger.debug(f"Rollout stopped at iteration {i+1}: Invalid action applied.")
                    print(f"Rollout stopped at iteration {i+1}: Invalid action applied.")
                    break
                current_state = new_state
                logger.debug(f"Rollout iteration {i+1}: Applied action {action}, new state: {current_state}")
                print(f"Rollout iteration {i+1}: Applied action {action}, new state: {current_state}")
            
            score = evaluate(current_state, model, protein_data)
            logger.debug(f"Rollout complete. Final state: {current_state}, Score: {score}")
            print(f"Rollout complete. Final state: {current_state}, Score: {score}")
            return score
        except Exception as e:
            logger.error(f"Error during rollout: {e}")
            print(f"Error during rollout: {e}")
            return 0.0

    def expand(self) -> 'MCTSNode':

        if not self.untried_actions:
            logger.debug("No untried actions to expand.")
            print("No untried actions to expand.")
            return self  # Return self if no untried actions are available
        try:
            action = self.untried_actions.pop()
            new_state = apply_action(self.state, action)
            if new_state is None:
                logger.debug(f"Expanded action resulted in invalid state: {action}")
                print(f"Expanded action resulted in invalid state: {action}")
                return self  # Return self if the action results in an invalid state
            child_node = MCTSNode(new_state, self.protein_sequence, parent=self)
            self.children.append(child_node)
            logger.debug(f"Expanded new child with SMILES: {new_state}")
            print(f"Expanded new child with SMILES: {new_state}")
            return child_node
        except Exception as e:
            logger.error(f"Error during node expansion: {e}")
            print(f"Error during node expansion: {e}")
            return self

    def backpropagate(self, result: float):
        self.visits += 1
        self.value += result
        logger.debug(f"Backpropagating result: {result} to node with SMILES: {self.state}")
        print(f"Backpropagating result: {result} to node with SMILES: {self.state}")
        if self.parent:
            self.parent.backpropagate(result)

class LigandMCTS:

    def __init__(self, model: EnhancedTelomeraseGNN, protein_data: Data, num_simulations: int = 100):

        self.model = model
        self.protein_data = protein_data
        self.num_simulations = num_simulations

    def search(self, initial_state: str, protein_sequence: str) -> str:

        try:
            root = MCTSNode(initial_state, protein_sequence)
            if not root.untried_actions:
                logger.warning(f"Invalid initial state or no valid modifications: {initial_state}")
                print(f"Invalid initial state or no valid modifications: {initial_state}")
                return initial_state

            logger.info(f"Starting MCTS search from initial state: {initial_state}")
            print(f"Starting MCTS search from initial state: {initial_state}")

            for i in range(self.num_simulations):
                logger.debug(f"Simulation {i+1}/{self.num_simulations}")
                print(f"Simulation {i+1}/{self.num_simulations}")
                node = root
                while node.is_fully_expanded() and node.children:
                    node = node.best_child()
                
                if not node.is_fully_expanded():
                    node = node.expand()
                    if node == root:  # Expansion failed
                        continue
                
                score = node.rollout(self.model, self.protein_data)
                node.backpropagate(score)

            if not root.children:
                logger.warning("MCTS failed to find any valid modifications.")
                print("MCTS failed to find any valid modifications.")
                return initial_state

            best_child = max(root.children, key=lambda c: c.visits)
            logger.info(f"MCTS search completed. Best SMILES: {best_child.state}")
            print(f"MCTS search completed. Best SMILES: {best_child.state}")
            return best_child.state
        except Exception as e:
            logger.error(f"Error during MCTS search: {e}")
            print(f"Error during MCTS search: {e}")
            return initial_state

# Define a dictionary of common maximum valences for elements
MAX_VALENCES = {
    1: 1,   # H
    6: 4,   # C
    7: 3,   # N
    8: 2,   # O
    9: 1,   # F
    15: 5,  # P
    16: 6,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1   # I
}

def get_valid_modifications(mol: Chem.Mol, protein_sequence: str) -> List[Tuple[str, int, str]]:
    modifications = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        current_valence = atom.GetExplicitValence()
        max_valence = MAX_VALENCES.get(atomic_num, 4)  # Default to 4 if not in the dictionary
        
        if atom.GetNumImplicitHs() > 0 and current_valence < max_valence:
            modifications.append(('add_hydrogen', idx, 'H'))
        
        if atom.GetNumExplicitHs() > 0:
            modifications.append(('remove_hydrogen', idx, 'H'))
        
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(idx, n_idx)
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                if current_valence + 1 <= max_valence and neighbor.GetExplicitValence() + 1 <= MAX_VALENCES.get(neighbor.GetAtomicNum(), 4):
                    modifications.append(('change_bond', idx, 'DOUBLE'))
            elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                modifications.append(('change_bond', idx, 'SINGLE'))
    
    return modifications




def evaluate(smiles: str, model: EnhancedTelomeraseGNN, protein_data: Data) -> float:

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES: {smiles}")
            print(f"Invalid SMILES: {smiles}")
            return 0.0

        logger.debug("Converting molecule to graph data for evaluation.")
        print("Converting molecule to graph data for evaluation.")
        ligand_data = molecule_to_graph_data(mol)
        ligand_data = ligand_data.to(next(model.parameters()).device)

        logger.debug("Performing forward pass through the model.")
        print("Performing forward pass through the model.")
        telomerase_activity, compound_effectiveness, pchembl_value = model(
            ligand_data.x,
            ligand_data.edge_index,
            ligand_data.edge_attr,
            ligand_data.batch,
            ligand_data.node_type
        )

        # Example scoring function
        score = telomerase_activity.item() + compound_effectiveness.item() - pchembl_value.item()
        logger.debug(f"Evaluation score for SMILES {smiles}: {score}")
        print(f"Evaluation score for SMILES {smiles}: {score}")
        return score
    except Exception as e:
        logger.error(f"Error evaluating SMILES {smiles}: {e}")
        print(f"Error evaluating SMILES {smiles}: {e}")
        return 0.0

def optimize_ligand(initial_smiles: str, protein_sequence: str, model: EnhancedTelomeraseGNN, num_iterations: int = 10) -> str:
    try:
        logger.info(f"Starting ligand optimization. Initial SMILES: {initial_smiles}")
        print(f"Starting ligand optimization. Initial SMILES: {initial_smiles}")
        protein_data = protein_to_graph_data(protein_sequence)
        
        # Move protein data to the same device as the model
        device = next(model.parameters()).device
        protein_data = protein_data.to(device)
        
        mcts = LigandMCTS(model, protein_data)
        current_smiles = initial_smiles
        best_score = float('-inf')

        for i in range(num_iterations):
            logger.info(f"Optimization iteration {i+1}/{num_iterations}")
            print(f"Optimization iteration {i+1}/{num_iterations}")
            new_smiles = mcts.search(current_smiles, protein_sequence)
            if new_smiles == current_smiles:
                logger.warning(f"MCTS failed to find a valid modification in iteration {i+1}.")
                print(f"MCTS failed to find a valid modification in iteration {i+1}.")
                continue
            new_score = evaluate(new_smiles, model, protein_data)
            logger.info(f"Iteration {i+1} complete. New SMILES: {new_smiles}, Score: {new_score}")
            print(f"Iteration {i+1} complete. New SMILES: {new_smiles}, Score: {new_score}")
            if new_score > best_score:
                current_smiles = new_smiles
                best_score = new_score
                logger.info(f"New best score: {best_score}")
                print(f"New best score: {best_score}")

        logger.info(f"Optimization complete. Final SMILES: {current_smiles}, Best score: {best_score}")
        print(f"Optimization complete. Final SMILES: {current_smiles}, Best score: {best_score}")
        return current_smiles
    except Exception as e:
        logger.error(f"Error during ligand optimization: {e}")
        print(f"Error during ligand optimization: {e}")
        return initial_smiles

def optimize_ligand_smiles(initial_smiles: str, protein_sequence: str) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = r"C:\Users\wes\AutoMol-v2\automol\phase2\phase2b\GNN.pth"  # Make sure this path is correct
    print("checkpoint_path", checkpoint_path)
    # Default configuration
    config = {
        "num_node_features": 1,
        "num_edge_features": 1,
        "hidden_channels": 128,
        "num_heads": 8,
        "num_node_types": 54
    }
    print("config", config)
    # Try to load configuration from file
    try:
        with open('config.json', 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config.get("model", {}))
        logger.info("Loaded configuration from config.json")
    except FileNotFoundError:
        logger.warning("config.json not found. Using default configuration.")
    except json.JSONDecodeError:
        logger.warning("Error parsing config.json. Using default configuration.")

    # Load the model
    model = load_model(checkpoint_path, config, device)
    print("model", model)
    if model is None:
        logger.error("Failed to load the model.")
        return initial_smiles  # Return the initial SMILES if model loading fails

    optimized_smiles = optimize_ligand(initial_smiles, protein_sequence, model)
    if not optimized_smiles:
        logger.error("Optimization returned an empty SMILES.")
        print("Optimization returned an empty SMILES.")
        return initial_smiles

    # Evaluate the optimized ligand
    final_score = evaluate(optimized_smiles, model, protein_to_graph_data(protein_sequence).to(device))
    logger.info(f"Final score: {final_score}")
    print(f"Final score: {final_score}")

    return optimized_smiles