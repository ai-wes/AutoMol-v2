import torch
from policy import Policy

def load_model(checkpoint_path, device='cpu'):
    # Initialize the model
    model = Policy(
        pretrained_model_name='mrm8488/chEMBL_smiles_v1',
        hidden_size=256,
        action_size=421,
        dropout_p=0.2,
        fine_tune=False
    ).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()  # Set the model to evaluation mode
    return model



from rdkit import Chem
from mol_rl_agent_run import MoleculeEnv, REACTIONS, PATHWAY_SCORING_FUNCTIONS, select_action

def optimize_smiles(model, smiles_sequence, num_steps=10, device='cpu'):
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles_sequence)
    if mol is None:
        raise ValueError("Invalid SMILES sequence")
    
    # Initialize the environment with the molecule
    env = MoleculeEnv(curriculum_level=1.0, reactions=REACTIONS, pathway_scoring_functions=PATHWAY_SCORING_FUNCTIONS)
    env.set_molecule(mol)
    
    optimized_smiles = []
    
    for _ in range(num_steps):
        state_mol = env.current_mol.GetMol()
        action = select_action(model, state_mol, device=device)
        next_state, reward, done, _ = env.step(action)
        
        if done:
            break
        
        optimized_smiles.append(Chem.MolToSmiles(env.current_mol.GetMol()))
    
    return optimized_smiles