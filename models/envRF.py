from cv2 import norm
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time
from icecream import ic
from . import render_modules
from models.ise import ISE, RandISE
from models.ish import ISH, RandISH

class EnvRF(torch.nn.Module):
    def __init__(self, rand_n=64, std=5, num_dense_layers=5, num_app_layers=1, featureC=256) -> None:
        
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
            torch.nn.Linear(featureC, 3)
        )
        self.app_mlp = torch.nn.Sequential(
            torch.nn.Linear(featureC+app_aug_C, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_app_layers)], []),
        )
        torch.nn.init.constant_(self.app_mlp[-1].bias, 0)
        self.mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            gain = 0.2688 if m.weight.shape[1] > 200 else 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            
    def compute_densityfeature(self, inner_dir, inv_depth):

        roughness = torch.tensor(20.0, device=inner_dir.device)
        indata = [self.ise(inner_dir, roughness), inv_depth]
        mlp_in = torch.cat(indata, dim=-1)
        upper_feature = self.dense_mlp(mlp_in)
        density = torch.dense_final(upper_feature)
        return density, upper_feature

    def compute_appfeature(self, upper_feature, view_dir, inner_dir, inv_depth, roughness):
        indata = [upper_feature]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.app_mlp(mlp_in)
        rgb = torch.sigmoid(rgb)
        return rgb