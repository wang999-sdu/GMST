# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        D = torch.sum(adj, dim=1)
        D_inv_sqrt = torch.pow(D + 1e-9, -0.5)
        D_inv_sqrt = torch.diag(D_inv_sqrt)
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        x = torch.mm(adj_norm, x)
        return F.relu(self.fc(x))

class MultiViewGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiViewGCN, self).__init__()
        self.feature_gcn = SimpleGCNLayer(input_dim, hidden_dim)
        self.spatial_gcn = SimpleGCNLayer(input_dim, hidden_dim)
        self.attention = nn.Linear(2 * hidden_dim, 1)

    def forward(self, feature_x, spatial_x, adj):
        H_f = self.feature_gcn(feature_x, adj)
        H_s = self.spatial_gcn(spatial_x, adj)
        H_cat = torch.cat([H_f, H_s], dim=1)
        attention_weights = torch.sigmoid(self.attention(H_cat))
        H = attention_weights * H_f + (1 - attention_weights) * H_s
        return H

class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MaskedAutoencoder, self).__init__()
        self.gcn = MultiViewGCN(input_dim, hidden_dim)
        self.fc_encoder = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_decoder = nn.Linear(hidden_dim // 2, input_dim)

    def forward(self, x, adj):
        H = self.gcn(x, x, adj)
        encoded = self.fc_encoder(H)
        mask = torch.bernoulli(torch.full_like(encoded, 0.75)).to(device)
        encoded = encoded * mask
        decoded = self.fc_decoder(encoded)
        return decoded