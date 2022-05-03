from .tensoRF import TensorVMSplit
from .tensor_nerf import TensorBase
import torch
import torch.nn.functional as F

class MultiLevelRF(TensorBase):
    def __init__(self, aabb, gridSize, device, num_levels, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)
        self.num_levels = num_levels
        self.res_divs = [(2**i) for i in range(num_levels)]
        self.gridSizes = [gridSize/ (2**i) for i in range(num_levels)]
        self.levels = [TensorVMSplit(aabb, gs, device, **kargs) for gs in self.gridSizes]

    def init_svd_volume(self, res, device):
        for level in self.levels:
            level.init_svd_volume(res, device)

    def compute_features(self, xyz_sampled):
        for level in self.levels:
            level.compute_features(xyz_sampled)
    
    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        grad_vars = []
        for level in self.levels:
            grad_vars += level.get_optparam_groups(lr_init_spatial, lr_init_network)
        return grad_vars
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass

    def shrink(self, new_aabb):
        for level in self.levels:
            level.shrink(new_aabb)

    def vector_comp_diffs(self):
        v = self.levels[0].vector_comp_diffs()
        for level in self.levels[1:]:
            v += level.vector_comp_diffs()
        return v

    def upsample_volume_grid(self, res_target):
        for level, res_div in zip(self.levels, self.res_divs):
            level.upsample_volume_grid(res_target / res_div)