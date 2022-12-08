import torch
import torch.nn as nn
import torch.nn.functional as F
from .sh import eval_sh_bases
from math import pi
from icecream import ic
from . import safemath
import numpy as np
from models.ise import ISE, RandISE
from models.ish import ISH, RandISH
import tinycudann as tcnn

def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

class Envmap(torch.nn.Module):
    def __init__(self, app_dim, nSamples=100) -> None:
        super().__init__()
        self.app_dim = app_dim
        self.rf = None
        self.nSamples = nSamples
        
    def normalize_coords(self, xyz):
        dist = torch.linalg.norm(xyz[..., :3], dim=-1, keepdim=True)
        direction = xyz[..., :3] / dist
        contracted_dist = torch.where(dist > 1, (2-1/dist), dist)
        return torch.cat([direction, contracted_dist, xyz[..., 3:]], dim=-1)

    # def recover_envmap(self, res):
            
    #     B = 2*res*res
    #     ele_grid, azi_grid = torch.meshgrid(
    #         torch.linspace(-np.pi/2, np.pi/2, res, dtype=torch.float32),
    #         torch.linspace(-np.pi, np.pi, 2*res, dtype=torch.float32), indexing='ij')
    #     # each col of x ranges from -pi/2 to pi/2
    #     # each row of y ranges from -pi to pi
    #     ang_vecs = torch.stack([
    #         torch.cos(ele_grid) * torch.cos(azi_grid),
    #         torch.cos(ele_grid) * torch.sin(azi_grid),
    #         -torch.sin(ele_grid),
    #     ], dim=-1).to(self.device)
    #     depth = torch.linspace(
    #     # roughness = 1/np.pi*torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    #     roughness = 20*torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    #     envmap = self.renderModule(xyz_samp, staticdir, app_features, refdirs=ang_vecs.reshape(-1, 3), roughness=roughness).reshape(res, 2*res, 3)
    #     color = self.renderModule(xyz_samp, ang_vecs.reshape(-1, 3), app_features, refdirs=staticdir, roughness=roughness).reshape(res, 2*res, 3)
    #     return envmap, color

class HashEnvmap(Envmap):
    def __init__(self, app_dim, nSamples=100, rand_n=128, std=5, num_dense_layers=5, num_app_layers=1, featureC=256) -> None:
        super().__init__(app_dim=app_dim, nSamples=nSamples)
        
        self.ise = RandISH(rand_n, std)
        
        dense_aug_C = 0
        app_aug_C = 0

        self.encoding = tcnn.Encoding(3, dict(
            otype="HashGrid",
            n_levels=16,
            n_features_per_level=2,
            log2_hashmap_size=19,
            base_resolution=2,
            per_level_scale=2
        ))
        self.rgb_network = tcnn.Network(self.encoding.n_output_dims+app_aug_C, app_dim, dict(
            otype="FullyFusedMLP",
            activation="ReLU",
            output_activation="None",
            n_neurons=64,
            n_hidden_layers=2,
        ))
        self.dense_network = tcnn.Network(self.encoding.n_output_dims, 1, dict(
            otype="FullyFusedMLP",
            activation="ReLU",
            output_activation="None",
            n_neurons=64,
            n_hidden_layers=2,
        ))

    def compute_densityfeature(self, xyz_env_normed):
        inner_dir = xyz_env_normed[..., :3]
        inv_depth = xyz_env_normed[..., 3:4]
        roughness = xyz_env_normed[..., 4:5]
        B = inner_dir.shape[0]
        
        x = inner_dir * inv_depth
        upper_feature = self.encoding(x)
        x = self.dense_network(upper_feature)

        return x.float().squeeze(-1), upper_feature.float()

    def compute_appfeature(self, upper_feature, xyz_env_normed):
        indata = [upper_feature]
        mlp_in = torch.cat(indata, dim=-1)
        app_feat = self.rgb_network(mlp_in)
        return app_feat.float()

class NeuralEnvmap(Envmap):
    def __init__(self, app_dim, nSamples=100, rand_n=128, std=5, num_dense_layers=5, num_app_layers=1, featureC=256) -> None:
        super().__init__(app_dim=app_dim, nSamples=nSamples)
        
        self.ise = RandISH(rand_n, std)
        
        dense_aug_C = 0
        app_aug_C = 0

        self.dense_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.ise.dim() + 1 + dense_aug_C, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_dense_layers)], []),
        )
        self.dense_final = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 1)
        )
        self.app_mlp = torch.nn.Sequential(
            torch.nn.Linear(featureC+app_aug_C, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_app_layers)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, app_dim),
        )
        self.dense_mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            gain = 0.2688 if m.weight.shape[1] > 200 else 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            
    def compute_densityfeature(self, xyz_env_normed):
        inner_dir = xyz_env_normed[..., :3]
        inv_depth = xyz_env_normed[..., 3:4]
        roughness = xyz_env_normed[..., 4:5]
        B = inner_dir.shape[0]

        indata = [self.ise(inner_dir, roughness).reshape(B, -1), inv_depth]
        mlp_in = torch.cat(indata, dim=-1)
        upper_feature = self.dense_mlp(mlp_in)
        density = self.dense_final(upper_feature)
        return density.squeeze(-1), upper_feature

    def compute_appfeature(self, upper_feature, xyz_env_normed):
        indata = [upper_feature]
        mlp_in = torch.cat(indata, dim=-1)
        app_feat = self.app_mlp(mlp_in)
        return app_feat