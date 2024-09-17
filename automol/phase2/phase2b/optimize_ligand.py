import pandas as pd
    # Target values
    
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from rdkit import Chem
from typing import List, Tuple
import logging
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import random
from typing import List, Tuple
import logging
from torch_geometric.data import Data, DataLoader, HeteroData



from collections import OrderedDict

# Extended GNN model for protein-ligand interaction prediction
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import random
from typing import List, Tuple

import torch
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
import torch.nn.functional as F

from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch

from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
import torch




class EnhancedTelomeraseGNN(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_heads, num_node_types):
        super(EnhancedTelomeraseGNN, self).__init__()
        self.node_embedding = nn.Embedding(num_node_types, hidden_channels)
        self.edge_embedding = nn.Linear(1, hidden_channels)  # Input features set to 1

        # Concatenated input dimension
        input_dim = num_node_features + hidden_channels  # Should be 1 + 128 = 129

        self.conv1 = GATConv(
            in_channels=input_dim,
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor, node_type: torch.Tensor):
        edge_attr = edge_attr.view(-1, 1)  # Ensure edge attributes have shape [num_edges, 1]

        node_embed = self.node_embedding(node_type)
        x = torch.cat([x, node_embed], dim=-1)


        edge_embed = self.edge_embedding(edge_attr)
        # Debug prints
        print(f"x shape after concatenation: {x.shape}")  # Should be [num_nodes, 129]
        print(f"edge_embed shape: {edge_embed.shape}")    # Should be [num_edges, hidden_channels]

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
        
        
# Function to load the model
def load_model(checkpoint_path: str, config: dict, device: torch.device):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    # Extract configuration parameters
    hidden_channels = config.get('hidden_channels', checkpoint.get('hidden_channels', 128))
    num_heads = config.get('num_heads', checkpoint.get('num_heads', 8))
    num_node_types = config.get('num_node_types', checkpoint.get('num_node_types', 54))

    # Set num_node_features and num_edge_features to match the checkpoint
    num_node_features = 1  # Ensure this matches the checkpoint
    num_edge_features = 1  # Ensure this matches the checkpoint

    # Initialize the model
    model = EnhancedTelomeraseGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_channels=hidden_channels,  # Should be 128 as per your checkpoint
        num_heads=num_heads,              # Should be 8 as per your checkpoint
        num_node_types=num_node_types     # Should be 54 as per your checkpoint
    ).to(device)
    # Load the state dictionary
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded and set to evaluation mode")
    except Exception as e:
        print(f"Error loading state dict into the model: {e}")
        return None

    model.eval()  # Set the model to evaluation mode
    return model


def process_activity_data(file_path: str) -> List[Data]:
    logger.info(f"Processing activity data from {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

    data_list = []

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol is not None:
            # Node features: Only one feature per node
            node_features = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)
            node_type = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
            edge_index = torch.tensor(
                [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()] +
                [[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] for b in mol.GetBonds()],
                dtype=torch.long
            ).t().contiguous()
            edge_attr = torch.tensor(
                [[b.GetBondTypeAsDouble()] for b in mol.GetBonds()] * 2,  # Include reverse edges
                dtype=torch.float
            )
            y = torch.tensor(
                [row['telomerase_activity'], row['compound_effectiveness'], row['pchembl_value']],
                dtype=torch.float
            )

            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_type=node_type,
                y=y
            )
            data_list.append(data)

        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1} molecules")

    logger.info(f"Finished processing {len(data_list)} molecules from activity data.")
    return data_list



def print_state_dict_structure(state_dict):
    print("State dict structure:")
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")
        elif isinstance(value, (dict, OrderedDict)):
            print(f"  {key}: (nested dictionary)")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"    {sub_key}: shape {sub_value.shape}")
                else:
                    print(f"    {sub_key}: type {type(sub_value)}")
        else:
            print(f"  {key}: type {type(value)}")

def smiles_to_mol(smiles: str) -> Chem.Mol:
    return Chem.MolFromSmiles(smiles)

def mol_to_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)




def process_molecule_data(smiles, telomerase_activity=None, compound_effectiveness=None, pchembl_value=None):
    logging.debug(f"Processing molecule data for SMILES: {smiles}")
    smiles_parts = smiles.split('.')
    if len(smiles_parts) > 1:
        logging.warning(f"SMILES contains multiple components. Processing the largest component.")
        smiles = max(smiles_parts, key=len)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"Unable to parse SMILES: {smiles}")
        return None

    # Node features: Only one feature per node
    x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)
    
    # Node types (atomic numbers)
    node_type = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    
    # Edge indices
    edge_index = torch.tensor(
        [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()],
        dtype=torch.long
    ).t().contiguous()
    
    # Edge attributes: Only one feature per edge
    edge_attr = torch.tensor([[bond.GetBondTypeAsDouble()] for bond in mol.GetBonds()], dtype=torch.float)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type
    )
    return data

def protein_to_graph_data(protein_sequence):
    amino_acids = list(protein_sequence)
    x = torch.tensor([ord(aa) for aa in amino_acids], dtype=torch.long).view(-1, 1)
    edge_index = torch.tensor([[i, i+1] for i in range(len(amino_acids)-1)] + 
                              [[i+1, i] for i in range(len(amino_acids)-1)], dtype=torch.long).t().contiguous()
    
    edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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
                if current_valence + 1 <= max_valence:
                    modifications.append(('change_bond', idx, 'DOUBLE'))
            elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                modifications.append(('change_bond', idx, 'SINGLE'))
    
    logging.debug(f"Valid modifications: {modifications}")
    return modifications


class MCTSNode:
    def __init__(self, state, protein_sequence, parent=None):
        self.state = state
        self.protein_sequence = protein_sequence
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        mol = Chem.MolFromSmiles(state)
        if mol is None:
            logging.warning(f"Invalid SMILES: {state}")
            self.untried_actions = []
        else:
            self.untried_actions = get_valid_modifications(mol, protein_sequence)

    # ... rest of the class remains the same
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        choices_weights = [
            (c.value / (c.visits + 1e-5)) + c_param * np.sqrt((2 * np.log(self.visits) / (c.visits + 1e-5)))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout(self, model, protein_data):
        current_state = self.state
        logging.debug(f"Starting rollout from state: {current_state}")
        for i in range(5):
            mol = Chem.MolFromSmiles(current_state)
            if not mol or Descriptors.MolWt(mol) > 500:
                logging.debug(f"Rollout stopped at iteration {i}: Invalid molecule or MW > 500")
                break
            possible_moves = get_valid_modifications(mol, self.protein_sequence)
            if not possible_moves:
                logging.debug(f"Rollout stopped at iteration {i}: No valid moves")
                break
            action = random.choice(possible_moves)
            new_state = apply_action(current_state, action)
            if new_state is None:
                logging.debug(f"Rollout stopped at iteration {i}: Invalid action")
                break
            current_state = new_state
            logging.debug(f"Rollout iteration {i}: Applied action {action}, new state: {current_state}")
        
        score = evaluate(current_state, model, protein_data)
        logging.debug(f"Rollout complete. Final state: {current_state}, Score: {score}")
        return score

    def expand(self):
        if not self.untried_actions:
            return self  # Return self if no untried actions are available
        action = self.untried_actions.pop()
        new_state = apply_action(self.state, action)
        if new_state is None:
            return self  # Return self if the action results in an invalid state
        child_node = MCTSNode(new_state, self.protein_sequence, parent=self)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)    
    
class LigandMCTS:
    def __init__(self, model, protein_data, num_simulations=100):
        self.model = model
        self.protein_data = protein_data
        self.num_simulations = num_simulations

    def search(self, initial_state, protein_sequence):
        root = MCTSNode(initial_state, protein_sequence)
        if not root.untried_actions:
            logging.warning(f"Invalid initial state or no valid modifications: {initial_state}")
            return initial_state

        logging.info(f"Starting MCTS search from initial state: {initial_state}")

        for i in range(self.num_simulations):
            logging.debug(f"Simulation {i+1}/{self.num_simulations}")
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
            logging.warning("MCTS failed to find any valid modifications")
            return initial_state

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.state    
    
    
def apply_action(smiles: str, action: Tuple[str, int, str]) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"Invalid SMILES: {smiles}")
        return None

    action_type, atom_idx, value = action
    
    try:
        atom = mol.GetAtomWithIdx(atom_idx)
        if action_type == 'add_hydrogen':
            # Add an explicit hydrogen
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
        elif action_type == 'remove_hydrogen':
            # Remove an explicit hydrogen
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
        logging.warning(f"Error applying action {action} to SMILES {smiles}: {str(e)}")
        return None
    
    
    
def evaluate(smiles: str, model: EnhancedTelomeraseGNN, protein_data) -> float:
    logging.debug(f"Evaluating SMILES: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logging.warning(f"Invalid molecule: {smiles}")
        return -1000  # Return a large negative value instead of -inf
    
    ligand_data = process_molecule_data(smiles)
    if ligand_data is None:
        logging.warning(f"Failed to process molecule data for SMILES: {smiles}")
        return -1000  # Return a large negative value instead of -inf

    # Move ligand data to the same device as the model
    device = next(model.parameters()).device
    ligand_data = ligand_data.to(device)
    protein_data = protein_data.to(device)

    ligand_batch = Batch.from_data_list([ligand_data])
    
    with torch.no_grad():
        # Check if node_type is available, if not, use a default value or skip it
        if hasattr(ligand_batch, 'node_type'):
            telomerase_activity, compound_effectiveness, pchembl_value = model(
                ligand_batch.x, 
                ligand_batch.edge_index, 
                ligand_batch.edge_attr, 
                ligand_batch.batch, 
                ligand_batch.node_type
            )
        else:
            logging.warning("node_type not available, using default values")
            telomerase_activity, compound_effectiveness, pchembl_value = model(
                ligand_batch.x, 
                ligand_batch.edge_index, 
                ligand_batch.edge_attr, 
                ligand_batch.batch, 
                torch.zeros(ligand_batch.x.size(0), dtype=torch.long, device=device)
            )
    score = telomerase_activity.item() + compound_effectiveness.item() + pchembl_value.item()
    logging.debug(f"Evaluation complete. Score: {score}")
    return score



def is_valid_action(mol, action):
    """
    Check if the proposed action is chemically valid.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): The current molecule.
        action (tuple): The proposed action (action_type, atom_idx, value).
    
    Returns:
        bool: True if the action is valid, False otherwise.
    """
    action_type, atom_idx, value = action
    
    if action_type == 'change_bond':
        atom = mol.GetAtomWithIdx(atom_idx)
        current_valence = atom.GetExplicitValence()
        max_valence = Chem.GetPeriodicTable().GetMaxValence(atom.GetAtomicNum())
        
        # Check if changing the bond would exceed max valence
        if value == 'DOUBLE' and current_valence + 1 > max_valence:
            return False
        elif value == 'TRIPLE' and current_valence + 2 > max_valence:
            return False
    
    elif action_type == 'add_hydrogen':
        atom = mol.GetAtomWithIdx(atom_idx)
        current_valence = atom.GetExplicitValence()
        max_valence = Chem.GetPeriodicTable().GetMaxValence(atom.GetAtomicNum())
        
        # Check if adding hydrogen would exceed max valence
        if current_valence + 1 > max_valence:
            return False
    
    return True


def optimize_ligand(initial_smiles: str, protein_sequence: str, model: EnhancedTelomeraseGNN, num_iterations: int = 10) -> str:
    logging.info(f"Starting ligand optimization. Initial SMILES: {initial_smiles}")
    protein_data = protein_to_graph_data(protein_sequence)
    
    # Move protein data to the same device as the model
    device = next(model.parameters()).device
    protein_data = protein_data.to(device)
    
    mcts = LigandMCTS(model, protein_data)
    current_smiles = initial_smiles
    best_score = float('-inf')

    for i in range(num_iterations):
        logging.info(f"Optimization iteration {i+1}/{num_iterations}")
        new_smiles = mcts.search(current_smiles, protein_sequence)
        if new_smiles == current_smiles:
            logging.warning(f"MCTS failed to find a valid modification in iteration {i+1}")
            continue
        new_score = evaluate(new_smiles, model, protein_data)
        logging.info(f"Iteration {i+1} complete. New SMILES: {new_smiles}, Score: {new_score}")
        if new_score > best_score:
            current_smiles = new_smiles
            best_score = new_score
            logging.info(f"New best score: {best_score}")

    logging.info(f"Optimization complete. Final SMILES: {current_smiles}, Best score: {best_score}")
    return current_smiles

def optimize_ligand_smiles(initial_smiles, protein_sequence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "GNN.pth"  # Update this to your actual checkpoint path

    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load the model
    model = load_model(checkpoint_path, config, device)
    if model is None:
        logger.error("Failed to load the model.")
        return

    optimized_smiles = optimize_ligand(initial_smiles, protein_sequence, model)
    # Optimize the ligand using the model
    logger.info(f"Optimization complete. Initial SMILES: {initial_smiles}")
    logger.info(f"Optimized SMILES: {optimized_smiles}")

    # Evaluate the optimized ligand
    final_score = evaluate(optimized_smiles, model, protein_to_graph_data(protein_sequence).to(device))
    logger.info(f"Final score: {final_score}")

    return optimized_smiles
