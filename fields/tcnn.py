import torch
import tinycudann as tcnn
from .tensor_base import TensorBase
import numpy as np
from icecream import ic
from mutils import normalize

class TCNNRF(TensorBase):
    def __init__(self, aabb, encoder_conf, grid_size, featureC=128, num_layers=4, tint_offset=0, diffuse_offset=-1, **kwargs):
        super().__init__(aabb, **kwargs)

        # self.nSamples = 1024                                                                                                                                                                                        
        # self.nSamples = 512                                                                                                                                                                                        
        self.nSamples = 512                                                                                                                                                                                        
        diag = (aabb**2).sum().sqrt()
        self.stepSize = diag / self.nSamples
        g = self.nSamples
        self.grid_size = torch.tensor([g, g, g])
        self.units = self.stepSize
        self.tint_offset = tint_offset
        self.diffuse_offset = diffuse_offset

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
            torch.nn.Linear(featureC, 14)
        )

    def get_optparam_groups(self):
        grad_vars = [
            {'params': self.encoding.parameters(), 'lr': self.lr},
            {'params': self.sigma_net.parameters(), 'lr': self.lr_net},
        ]
        return grad_vars

    def density_L1(self):
        return torch.tensor(0.0, device=self.get_device())

    def check_schedule(self, iter, batch_mul):
        return False

    def coords2input(self, xyz_normed):
        return (xyz_normed[..., :3].reshape(-1, 3)/2+0.5).contiguous()

    def compute_feature(self, xyz_normed):
        feat = self.encoding(self.coords2input(xyz_normed)).type(xyz_normed.dtype)
        mlp_out = self.sigma_net(feat)
        sigfeat = mlp_out[:, 0]
        normals = normalize(mlp_out[:, 1:4])
        mlp_out = mlp_out[:, 4:]

        r1 = torch.sigmoid(mlp_out[..., 7:8]).clip(min=1e-2)
        r2 = torch.sigmoid(mlp_out[..., 8:9]).clip(min=1e-2)
        # ic(mlp_out[..., 0:6])
        tint = torch.sigmoid((mlp_out[..., 3:6]+self.tint_offset).clip(min=-10, max=10))
        # ic(tint.mean())
        f0 = (torch.sigmoid((mlp_out[..., 9:10]+3).clip(min=-10, max=10))+0.001).clip(max=1)
        # diffuse = rgb[..., :3]
        # tint = F.softplus(mlp_out[..., 3:6])
        diffuse = torch.sigmoid((mlp_out[..., :3]+self.diffuse_offset))

        return self.feature2density(sigfeat).reshape(-1), normals, feat, dict(
            diffuse = diffuse,
            r1 = r1,
            r2 = r2,
            f0 = f0,
            tint=tint,
        )

    def compute_appfeature(self, xyz_normed):
        feat = self.encoding(xyz_normed[..., :3].reshape(-1, 3).contiguous()).type(xyz_normed.dtype)
        return feat

    def compute_densityfeature(self, xyz_normed, activate=True):
        feat = self.encoding(self.coords2input(xyz_normed)).type(xyz_normed.dtype)
        mlp_out = self.sigma_net(feat)
        sigfeat = mlp_out[:, 0]
        if activate:
            return self.feature2density(sigfeat).reshape(-1)
        else:
            return sigfeat.reshape(-1)

    def shrink(self, new_aabb, voxel_size):
        pass
