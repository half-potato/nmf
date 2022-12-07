import torch
import tinycudann as tcnn
from .tensor_base import TensorBase
import numpy as np
from icecream import ic
from mutils import normalize
import torch.nn.functional as F
from models import util
import math
from models.grid_sample_Cinf import grid_sample
from fields.tensoRF import TensoRF

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

class HybridRF(TensorBase):
    def __init__(self, aabb, base_resolution, max_resolution, enc_dim, n_features_per_level, grid_levels, n_levels, lr_density,
                 encoder_conf, grid_conf, TV_samples, TV_scale, roughness_bias=-1, tint_offset=0, diffuse_offset=-1, enc_mul=1,
                 **kwargs):
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
        self.TV_scale = TV_scale
        self.TV_samples = TV_samples
        self.lr_density = lr_density

        self.separate_appgrid = False

        self.n_features_per_level = n_features_per_level
        self.n_levels = n_levels
        self.grid_levels = grid_levels

        self.bound = torch.abs(aabb).max()
        bound = 1
        self.per_level_scale = np.exp2(np.log2(max_resolution * bound / base_resolution) / (grid_levels + n_levels))
        ic(self.per_level_scale, base_resolution * self.per_level_scale**grid_levels)

        # self.grid = grid(base_resolution=base_resolution, dim=grid_levels*encoder_conf.n_features_per_level)
        # self.grid = grid(base_resolution=base_resolution, dim=grid_levels*encoder_conf.n_features_per_level)

        self.grid = tcnn.Encoding(3, encoding_config=dict(n_levels=grid_levels, base_resolution=base_resolution, per_level_scale=self.per_level_scale,
                                                          n_features_per_level=self.n_features_per_level, **grid_conf))
        self.encoding = tcnn.Encoding(3, encoding_config=dict(n_levels=n_levels, base_resolution=base_resolution * self.per_level_scale**grid_levels,
                                                              n_features_per_level=self.n_features_per_level, per_level_scale=self.per_level_scale, **encoder_conf))
        torch.nn.init.uniform_(list(self.grid.parameters())[0], -1e-2, 1e-2)
        # torch.nn.init.constant_(list(self.grid.parameters())[0], 1e-3)
        # ic(list(self.grid.parameters())[0])
        # self.grid.apply()
        app_dim = n_features_per_level * (grid_levels)


        self.sigma_net = util.create_mlp(app_dim, enc_dim, **kwargs)
        self.density_layer = util.create_mlp(enc_dim, 1, 1, initializer='kaiming')
        self.app_dim = enc_dim

    def get_optparam_groups(self, lr_scale=1):
        grad_vars = [
            {'params': self.encoding.parameters(), 'lr': self.lr*lr_scale},
            {'params': self.sigma_net.parameters(), 'lr': self.lr_net*lr_scale},
            {'params': self.density_layer.parameters(), 'lr': self.lr_density*lr_scale},
            {'params': self.grid.parameters(), 'lr': self.lr*lr_scale},
        ]
        ic(list(self.grid.parameters())[0].shape)
        return grad_vars

    def density_L1(self):
        return torch.tensor(0.0, device=self.get_device())

    def check_schedule(self, iter, batch_mul):
        return False

    def coords2input(self, xyz_normed):
        return (xyz_normed[..., :3].reshape(-1, 3)/2+0.5).contiguous()

    def calc_feat(self, xyz_normed):
        ifeat = self.encoding(self.coords2input(xyz_normed)).type(xyz_normed.dtype)
        gfeat = self.grid(self.coords2input(xyz_normed)).type(xyz_normed.dtype)
        # tfeat = self.grid(xyz_normed)
        # tfeat = sum(tfeat).T
        # tfeat = torch.cat(tfeat, dim=0).T
        # feat = torch.cat([gfeat, ifeat], dim=-1)
        feat = gfeat
        # device = xyz_normed.device
        # scale = self.per_level_scale**torch.arange(0, self.grid_levels, device=device).repeat_interleave(self.n_features_per_level)
        # var = xyz_normed[:, 3]
        # feat_scale = torch.exp(-(scale**2).reshape(1, -1) * var.reshape(-1, 1))
        # feat = feat_scale * feat * self.enc_mul
        feat = feat * self.enc_mul
        h = self.sigma_net(feat)
        # sigfeat = h[:, 0]
        # h = h[:, 1:]
        sigfeat = self.density_layer(h)
        # sigfeat = sum(sigfeat).sum(dim=0)

        return sigfeat, h

    def _compute_feature(self, xyz_normed):
        sigfeat, h = self.calc_feat(xyz_normed)
        return self.feature2density(sigfeat).reshape(-1), h

    def _compute_appfeature(self, xyz_normed):
        sigfeat, h = self.calc_feat(xyz_normed)
        return h

    def _compute_densityfeature(self, xyz_normed):
        sigfeat, h = self.calc_feat(xyz_normed)
        return sigfeat

    def shrink(self, new_aabb, voxel_size):
        pass

    def TV_loss_density(self, *args):
        # TV_scale is the width of the cube
        # sample cube center between -1 and 1 that doesn't intersect the walls
        device = self.get_device()
        center = (torch.rand(3, device=device)*2-1) * (1-self.TV_scale/2)
        rng = torch.linspace(-self.TV_scale/2, self.TV_scale/2, self.TV_samples, device=device)
        X, Y, Z = torch.meshgrid(rng, rng, rng, indexing='xy')
        coords = torch.stack([X + center[0], Y + center[1], Z + center[2], torch.zeros_like(X)], dim=-1)
        d = self.compute_densityfeature(coords.reshape(-1, 4)).reshape(coords.shape[:-1])
        h_tv = d[1:, :-1, :-1] - d[:-1, :-1, :-1]
        w_tv = d[:-1, 1:, :-1] - d[:-1, :-1, :-1]
        d_tv = d[:-1, :-1, 1:] - d[:-1, :-1, :-1]
        return (h_tv.abs() + w_tv.abs() + d_tv.abs()).mean()
        # return (h_tv**2 + w_tv**2 + d_tv**2 + torch.finfo(torch.float32).eps).sqrt().mean()
