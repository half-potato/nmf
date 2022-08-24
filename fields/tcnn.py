import torch
import tinycudann as tcnn
from .tensor_base import TensorBase
import numpy as np
from icecream import ic

class TCNNRF(TensorBase):
    def __init__(self, aabb, encoder_conf, featureC=128, num_layers=4, **kwargs):
        super().__init__(aabb, **kwargs)

        # self.nSamples = 1024                                                                                                                                                                                        
        # self.nSamples = 512                                                                                                                                                                                        
        self.nSamples = 512                                                                                                                                                                                        
        diag = (aabb**2).sum().sqrt()
        self.stepSize = diag / self.nSamples
        g = self.nSamples
        self.grid_size = torch.tensor([g, g, g])
        self.units = self.stepSize


        self.separate_appgrid = False

        self.bound = torch.abs(aabb).max()
        bound = 1
        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))
        self.encoding = tcnn.Encoding(3, encoding_config=dict(per_level_scale=per_level_scale, **encoder_conf))
        self.app_dim = encoder_conf.n_features_per_level * encoder_conf.n_levels
        # self.sigma_net = tcnn.Network(n_input_dims=self.app_dim, n_output_dims=1, network_config=dict(**network_config))
        self.sigma_net = torch.nn.Sequential(
            torch.nn.Linear(self.app_dim, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_layers-2)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 1)
        )

    def get_optparam_groups(self):
        grad_vars = [
            {'params': self.encoding.parameters(), 'lr': self.lr},
            {'params': self.sigma_net.parameters(), 'lr': self.lr_net},
        ]
        return grad_vars

    def check_schedule(self, iter):
        return False

    def coords2input(self, xyz_normed):
        return (xyz_normed[..., :3].reshape(-1, 3)/2+0.5).contiguous()

    def compute_feature(self, xyz_normed):
        feat = self.encoding(self.coords2input(xyz_normed)).type(xyz_normed.dtype)
        sigfeat = self.sigma_net(feat)
        return self.feature2density(sigfeat).reshape(-1), feat

    def compute_appfeature(self, xyz_normed):
        feat = self.encoding(xyz_normed[..., :3].reshape(-1, 3).contiguous()).type(xyz_normed.dtype)
        return feat

    def compute_densityfeature(self, xyz_normed):
        feat = self.encoding(self.coords2input(xyz_normed)).type(xyz_normed.dtype)
        sigfeat = self.sigma_net(feat)
        return self.feature2density(sigfeat).reshape(-1)

    def shrink(self, new_aabb, voxel_size):
        pass
