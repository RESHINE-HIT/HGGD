import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbors=16):
        super(PointTransformerLayer, self).__init__()
        self.num_neighbors = num_neighbors
        self.fc_q = nn.Linear(in_channels, out_channels)
        self.fc_k = nn.Linear(in_channels, out_channels)
        self.fc_v = nn.Linear(in_channels, out_channels)
        self.fc_gamma = nn.Linear(out_channels, out_channels)
        self.fc_out = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, pos):
        B, C, N = x.shape  # B: center-num, C: 35, N: 512

        # Transpose x to [B, N, C] for linear layers
        x = x.transpose(1, 2)  # [B, N, C]
        pos = pos.transpose(1, 2)  # [B, N, 3]
        # Compute queries, keys, and values
        q = self.fc_q(x)  # [B, N, out_channels]
        k = self.fc_k(x)  # [B, N, out_channels]
        v = self.fc_v(x)  # [B, N, out_channels]

        # Compute pairwise distances
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.sum(pos_diff ** 2, dim=-1)  # [B, N, N]

        # Find the nearest neighbors
        _, idx = torch.topk(dist, self.num_neighbors, dim=-1, largest=False, sorted=False)  # [B, N, num_neighbors]

        # Gather the neighbors
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, k.size(-1))  # [B, N, num_neighbors, out_channels]
        k_neighbors = torch.gather(k.unsqueeze(2).expand(-1, -1, N, -1), 2, idx_expanded)  # [B, N, num_neighbors, out_channels]
        v_neighbors = torch.gather(v.unsqueeze(2).expand(-1, -1, N, -1), 2, idx_expanded)  # [B, N, num_neighbors, out_channels]

        # Compute attention weights
        attn = torch.softmax(q.unsqueeze(2) - k_neighbors, dim=-1)  # [B, N, num_neighbors, out_channels]

        # Compute the output
        x = torch.sum(attn * v_neighbors, dim=-2)  # [B, N, out_channels]
        gamma = self.fc_gamma(x)  # [B, N, out_channels]
        x = self.fc_out(x + gamma)  # [B, N, out_channels]

        # Transpose x back to [B, out_channels, N]
        x = x.transpose(1, 2)  # [B, out_channels, N]
        return x

class PointTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=5, num_neighbors=16):
        super(PointTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.layers.append(PointTransformerLayer(in_ch, out_channels, num_neighbors))

    def forward(self, x, pos):
        for layer in self.layers:
            x = layer(x, pos)
        return x