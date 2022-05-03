from .tensoRF import TensorVMSplit
from .tensor_base import TensorBase
import torch
import torch.nn.functional as F
from icecream import ic

class MultiLevelRF(TensorBase):
    def __init__(self, aabb, gridSize, device, density_n_comp, appearance_n_comp, app_dim, step_ratio,
                 num_levels, *args, **kargs):
        self.num_levels = num_levels
        self.res_divs = [(2**i) for i in range(num_levels)]
        self.gridSizes = [[g / (2**i) for g in gridSize] for i in range(num_levels)]
        self.levels = [TensorVMSplit(aabb, gs, device, density_n_comp, appearance_n_comp, app_dim, step_ratio, *args, **kargs) for gs in self.gridSizes]
        super().__init__(aabb, gridSize, device, density_n_comp, appearance_n_comp, app_dim, step_ratio, *args, **kargs)

    def normalize_coord(self, xyz_sampled):
        return self.levels[0].normalize_coord(xyz_sampled)

    # @property
    # def invaabbSize(self):
    #     return self.levels[0].invaabbSize

    def density_L1(self):
        return sum([level.density_L1() for level in self.levels])
            

    def init_svd_volume(self, res, device):
        for level in self.levels:
            level.init_svd_volume(res, device)

    def update_stepSize(self, gridSize):
        print("MultiLevelRF: update_stepSize")
        super().update_stepSize(gridSize)
        for level, res_div in zip(self.levels, self.res_divs):
            level.update_stepSize([g/res_div for g in gridSize])

    def init_svd_volume(self, res, device):
        pass
   
    def compute_features(self, xyz_sampled):
        for level in self.levels:
            level.compute_features(xyz_sampled)
    
    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        grad_vars = []
        for level in self.levels:
            grad_vars += level.get_optparam_groups(lr_init_spatial, lr_init_network)
        return grad_vars
    
    def calculate_size_weights(self, xyz_sampled):
        size = xyz_sampled[..., 3]
        # the sum across the weights is 1
        # just want the closest grid point
        # first calculate the size of voxels in meters
        voxel_sizes = torch.tensor([level.units.max() for level in self.levels], dtype=torch.float32, device=xyz_sampled.device)

        # then, the weight should be summed from the smallest supported size to the largest
        pad = [1] * (len(size.shape)-1)
        size_diff = abs(voxel_sizes.reshape(1, len(self.levels)) - size.reshape(-1, 1))/voxel_sizes.max()
        weights = F.softmax(-size_diff.reshape(*size.shape, len(self.levels)), dim=-1)
        return weights
    
    def compute_densityfeature(self, xyz_sampled):
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        weights = self.calculate_size_weights(xyz_sampled)
        for i, level in enumerate(self.levels):
            sigma_feature += level.compute_densityfeature(xyz_sampled) * weights[..., i]
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        app_feature = torch.zeros((xyz_sampled.shape[0], self.app_dim), device=xyz_sampled.device)
        weights = self.calculate_size_weights(xyz_sampled)
        for i, level in enumerate(self.levels):
            app_feature += level.compute_appfeature(xyz_sampled) * weights[..., i, None]
        return app_feature

    def compute_density_norm(self, xyz_sampled):
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        normal_feature = torch.zeros((xyz_sampled.shape[0], 3), device=xyz_sampled.device)
        weights = self.calculate_size_weights(xyz_sampled)
        for i, level in enumerate(self.levels):
            sf, nf = level.compute_density_norm(xyz_sampled)
            sigma_feature += sf * weights[..., i]
            normal_feature += nf * weights[..., i, None]
        return sigma_feature, normal_feature

    def shrink(self, new_aabb, apply_correction):
        for level in self.levels:
            level.shrink(new_aabb, apply_correction)

    def vector_comp_diffs(self):
        v = self.levels[0].vector_comp_diffs()
        for level in self.levels[1:]:
            v += level.vector_comp_diffs()
        return v

    def upsample_volume_grid(self, res_target):
        for level, res_div in zip(self.levels, self.res_divs):
            level.upsample_volume_grid([g // res_div for g in res_target])