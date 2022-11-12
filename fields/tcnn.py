import torch
import tinycudann as tcnn
from .tensor_base import TensorBase
import numpy as np
from icecream import ic
from mutils import normalize
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        torch.nn.init.kaiming_uniform_(m.weight)

class TCNNRF(TensorBase):
    def __init__(self, aabb, encoder_conf, grid_size, enc_dim, roughness_bias=-1, featureC=128, num_layers=4, tint_offset=0, diffuse_offset=-1, **kwargs):
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
        self.roughness_bias = roughness_bias

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
            torch.nn.Linear(featureC, enc_dim)
        )
        self.tint_head = torch.nn.Linear(enc_dim, 3)
        self.diffuse_head = torch.nn.Linear(enc_dim, 3)
        self.roughness_head = torch.nn.Linear(enc_dim, 1)
        # self.normal_head = torch.nn.Linear(enc_dim, 3, bias=False)
        self.normal_head = torch.nn.Sequential(
            torch.nn.Linear(enc_dim, featureC),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Linear(hdim, hdim),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3, bias=False),
        )
        self.tint_head.apply(init_weights)
        self.diffuse_head.apply(init_weights)
        self.roughness_head.apply(init_weights)
        self.normal_head.apply(init_weights)

    def get_optparam_groups(self, lr_scale=1):
        grad_vars = [
            {'params': self.encoding.parameters(), 'lr': self.lr*lr_scale},
            {'params': self.sigma_net.parameters(), 'lr': self.lr_net*lr_scale},
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
        h = self.sigma_net(feat)
        sigfeat = h[:, 0]

        diffuse = torch.sigmoid(self.diffuse_head(h)+self.diffuse_offset)
        tint = torch.sigmoid(self.tint_head(h))
        roughness = F.softplus(self.roughness_head(h) + self.roughness_bias)
        raw_norms = self.normal_head(h)
        pred_norms = normalize(raw_norms)

        return self.feature2density(sigfeat).reshape(-1), pred_norms, feat, dict(
            diffuse = diffuse,
            r1 = roughness,
            r2 = roughness,
            tint=tint,
        )

    def compute_appfeature(self, xyz_normed):
        feat = self.encoding(xyz_normed[..., :3].reshape(-1, 3).contiguous()).type(xyz_normed.dtype)
        return feat

    def compute_densityfeature(self, xyz_normed, activate=True):
        feat = self.encoding(self.coords2input(xyz_normed)).type(xyz_normed.dtype)
        x = self.sigma_net(feat)
        sigfeat = x[:, 0]
        if activate:
            return self.feature2density(sigfeat).reshape(-1)
        else:
            return sigfeat.reshape(-1)

    def shrink(self, new_aabb, voxel_size):
        pass
