import torch
from icecream import ic
import torch.nn.functional as F
from mutils import normalize
from .tensor_base import TensorBase

class Ball(TensorBase):
    def __init__(self, radius=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # adjust radius
        self.radius = radius / self.aabb[0, 0]
        self.app_dim = 1
        self.nSamples = 512                                                                                                                                                                                        

        diag = (self.aabb**2).sum().sqrt()
        self.stepSize = diag / self.nSamples
        g = self.nSamples
        self.grid_size = torch.tensor([g, g, g])
        self.units = self.stepSize

    def get_optparam_groups(self):
        return {}

    def density_L1(self):
        return torch.tensor(0.0, device=self.get_device())

    def compute_feature(self, xyz_normed):
        return self.compute_densityfeature(xyz_normed), self.compute_appfeature(xyz_normed)

    def compute_densityfeature(self, xyz_sampled):
        return torch.where(torch.linalg.norm(xyz_sampled, dim=-1) < self.radius, 99999999.0, 0.0)

    def compute_appfeature(self, xyz_sampled):
        return torch.zeros_like(xyz_sampled[..., 0:1])

    def calculate_normals(self, xyz):
        return normalize(xyz[..., :3])
