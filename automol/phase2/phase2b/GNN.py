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