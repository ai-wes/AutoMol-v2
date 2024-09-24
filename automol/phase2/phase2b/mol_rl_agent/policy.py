


# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, pretrained_model_name='mrm8488/chEMBL_smiles_v1', hidden_size=256, action_size=40, dropout_p=0.2, fine_tune=False):
        super(Policy, self).__init__()
        # Load pretrained MLM model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
        self.pretrained_model = RobertaModel.from_pretrained(pretrained_model_name)
        self.pretrained_model.to(device)
        
        
        # Optionally fine-tune the pretrained model
        if fine_tune:
            for param in self.pretrained_model.parameters():
                param.requires_grad = True
        else:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # Additional layers for the policy network
        self.affine1 = nn.Linear(self.pretrained_model.config.hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Storage for actions and rewards
        self.saved_actions = []
        self.rewards = []
    
    def forward(self, smiles):
        """
        Forward pass of the Policy network.
        
        Args:
            smiles (list of str): List of SMILES strings.
        
        Returns:
            action_prob (Tensor): Probabilities over actions.
            state_value (Tensor): Estimated state value.
        """
        # Tokenize input SMILES
        inputs = self.tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=100).to(device)
        
        with torch.no_grad():
            outputs = self.pretrained_model(**inputs)
        
        # Use [CLS] token representation
        cls_representation = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through additional layers
        x = F.relu(self.affine1(cls_representation))
        x = self.dropout(x)
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        
        return action_prob, state_value
