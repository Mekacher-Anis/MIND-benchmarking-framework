import torch
import torch.nn as nn

class AdditiveAttention(torch.nn.Module):
    def __init__(self, in_dim=100, v_size=200):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        # self.v = torch.nn.Parameter(torch.rand(self.v_size))
        self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
        self.proj_v = nn.Linear(self.v_size, 1)

    def forward(self, context):
        """Additive Attention

        Args:
            context (tensor): [B, seq_len, in_dim]

        Returns:
            outputs, weights: [B, seq_len, out_dim], [B, seq_len]
        """
        # weights = self.proj(context) @ self.v
        weights = self.proj_v(self.proj(context)).squeeze(-1) # [B, seq_len]
        weights = torch.softmax(weights, dim=-1) # [B, seq_len]
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights # [B, 1, seq_len], [B, seq_len, dim]