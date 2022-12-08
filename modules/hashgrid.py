import torch
import torch.nn as nn
import torch.nn.functional as F
from .sh import eval_sh_bases
from math import pi
from icecream import ic
from . import safemath

class Sin(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class TrigHashGrid(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_per_level=2, level_dim=1000, max_freq=10, M=3) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.M = M
        self.out_dim = out_dim
        self.num_per_level = num_per_level
        self.level_dim = level_dim
        N = self.out_dim // self.num_per_level
        self.grids = nn.ParameterList([
            # nn.Parameter(0.5 * torch.rand((1, bg_rank, self.power**i * bg_resolution*unwrap_fn.H_mul, self.power**i * bg_resolution*unwrap_fn.W_mul)))
            nn.Parameter(0.1 * torch.ones((1, self.num_per_level, 1, level_dim)))
            for _ in range(N)])
        G = (torch.randn((in_dim, M, N))) * max_freq
        H = (torch.randn(M, N)) * max_freq
        self.register_buffer('G', G)
        self.register_buffer('H', H)
        featureC = 128
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, featureC),
            Sin(),
            # torch.nn.Linear(featureC, featureC),
            # Sin(),
            torch.nn.Linear(featureC, N),
            Sin(),
        )

    def dim(self):
        return self.out_dim

    def forward(self, x, size):
        B = x.shape[0]
        a = x @ self.G.reshape(self.in_dim, -1)
        gx = torch.sin(a.reshape(B, self.M, -1) + self.H.reshape(1, self.M, -1)).prod(dim=1)
        # gx = self.mlp(x)
        z = torch.zeros(B, device=x.device)

        embs = []
        for i, grid in enumerate(self.grids):
            index = torch.stack([gx[:, i], z], dim=-1).reshape(1, 1, -1, 2)
            emb = F.grid_sample(grid, index, mode='bicubic', align_corners=False)
            emb = emb.reshape(self.num_per_level, -1).T
            embs.append(emb)
        return torch.cat(embs, dim=-1)
