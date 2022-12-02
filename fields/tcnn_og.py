import torch
import tinycudann as tcnn
from .tensor_base import TensorBase
import numpy as np
from icecream import ic
from mutils import normalize
import torch.nn.functional as F
from models import util

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

class TCNNRF(TensorBase):
    def __init__(self, aabb, encoder_conf, grid_size, enc_dim, roughness_bias=-1, tint_offset=0, diffuse_offset=-1, enc_mul=1, **kwargs):
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
        self.enc_mul = enc_mul

        self.separate_appgrid = False

        self.bound = torch.abs(aabb).max()
        bound = 1
        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoding = tcnn.Encoding(3, encoding_config=dict(per_level_scale=per_level_scale, **encoder_conf))
        app_dim = encoder_conf.n_features_per_level * encoder_conf.n_levels
        self.n_features_per_level = encoder_conf.n_features_per_level
        self.n_levels = encoder_conf.n_levels
        # self.sigma_net = tcnn.Network(n_input_dims=self.app_dim, n_output_dims=1, network_config=dict(**network_config))
        self.sigma_net = util.create_mlp(app_dim, enc_dim, **kwargs)
        self.density_layer = util.create_mlp(enc_dim, 1, 1, initializer='kaiming')
        self.app_dim = enc_dim
        # self.sigma_net.apply(init_weights)

    def get_optparam_groups(self, lr_scale=1):
        grad_vars = [
            {'params': self.encoding.parameters(), 'lr': self.lr*lr_scale},
            {'params': self.sigma_net.parameters(), 'lr': self.lr_net*lr_scale},
            {'params': self.density_layer.parameters(), 'lr': self.lr_net*lr_scale},
        ]
        return grad_vars

    def density_L1(self):
        return torch.tensor(0.0, device=self.get_device())

    def check_schedule(self, iter, batch_mul):
        return False

    def coords2input(self, xyz_normed):
        return (xyz_normed[..., :3].reshape(-1, 3)/2+0.5).contiguous()

    def calc_feat(self, xyz_normed):
        feat = self.encoding(self.coords2input(xyz_normed)).type(xyz_normed.dtype)
        device = xyz_normed.device
        scale = 2**torch.arange(0, self.n_levels, device=device).repeat_interleave(self.n_features_per_level)
        # var = 0*xyz_normed[:, 2]+0.0001
        var = xyz_normed[:, 3]
        feat_scale = torch.exp(-(scale**2).reshape(1, -1) * var.reshape(-1, 1))
        feat = feat_scale * feat * self.enc_mul
        h = self.sigma_net(feat)
        # sigfeat = h[:, 0]
        # h = h[:, 1:]
        sigfeat = self.density_layer(h)

        # x = feat
        # for i, layer in enumerate(self.sigma_net.children()):
        #     x = layer(x)
        #     if hasattr(layer, 'weight') and layer.weight.grad is not None:
        #         ic(i, x[0], layer.weight.shape, layer.weight.mean(dim=0), layer.weight.grad.mean(dim=0))
        return sigfeat, h

    def _compute_feature(self, xyz_normed):
        sigfeat, h = self.calc_feat(xyz_normed)
        return self.feature2density(sigfeat).reshape(-1), h

    def _compute_appfeature(self, xyz_normed):
        sigfeat, h = self.calc_feat(xyz_normed)
        h = h[:, 1:]
        return h

    def _compute_densityfeature(self, xyz_normed, activate=True):
        sigfeat, h = self.calc_feat(xyz_normed)
        if activate:
            return self.feature2density(sigfeat).reshape(-1)
        else:
            return sigfeat.reshape(-1)

    def shrink(self, new_aabb, voxel_size):
        pass

