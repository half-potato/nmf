import torch
from icecream import ic
from typing import List

class TensorBase(torch.nn.Module):
    aabb: List[int]
    grid_size: List[int]
    density_n_comp: int
    appearance_n_comp: int
    app_dim: int
    step_ratio: float
    density_res_multi: float
    contract_space: bool
    hier_sizes: List[int]
    def __init__(self, aabb, grid_size, density_n_comp, appearance_n_comp,
                 app_dim, step_ratio, density_res_multi, contract_space):
        super().__init__()
        self.dtype = torch.half
        self.density_n_comp = [density_n_comp]*3
        self.app_n_comp = [appearance_n_comp]*3
        self.density_res_multi = density_res_multi
        self.app_dim = app_dim
        self.register_buffer('aabb', aabb)
        self.step_ratio = step_ratio
        self.contract_space = contract_space

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.update_stepSize(grid_size)
        self.init_svd_volume(grid_size[0])

    def get_kwargs(self):
        return {
            'grid_size':self.grid_size.tolist(),
            'aabb': self.aabb,
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,
            'step_ratio': self.step_ratio,
            'density_res_multi': self.density_res_multi,
        }
        
    def set_register(self, name, val):
        if hasattr(self, name):
            setattr(self, name, val.type_as(getattr(self, name)))
        else:
            self.register_buffer(name, val)

    def update_stepSize(self, grid_size):
        grid_size = torch.LongTensor(grid_size)
        print("grid size", grid_size)
        print("density grid size", [int(self.density_res_multi*g) for g in grid_size])
        print("aabb", self.aabb.view(-1))
        aabbSize = self.aabb[1] - self.aabb[0]
        self.set_register('invaabbSize', 2.0/aabbSize)
        self.set_register('grid_size', grid_size)
        self.set_register('units', aabbSize.to(self.grid_size.device) / (self.grid_size-1))
        # min is more accurate than mean
        self.set_register('stepSize', torch.min(self.units)*self.step_ratio)
        self.set_register('aabbDiag', torch.sqrt(torch.sum(torch.square(aabbSize))))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) * 2
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)
        
    def contract_coord(self, xyz_sampled): 
        dist = torch.linalg.norm(xyz_sampled[..., :3], dim=1, keepdim=True) + 1e-8
        direction = xyz_sampled[..., :3] / dist
        contracted = torch.where(dist > 1, (2-1/dist), dist) * direction
        return torch.cat([ contracted, xyz_sampled[..., 3:] ], dim=-1)

    def normalize_coord(self, xyz_sampled):
        coords = (xyz_sampled[..., :3]-self.aabb[0]) * self.invaabbSize - 1
        size = xyz_sampled[..., 3:4]
        normed = torch.cat((coords, size), dim=-1)
        if self.contract_space:
            r = 1
            d = 3
            dist = torch.linalg.norm(normed[..., :d], dim=-1, keepdim=True, ord=2) + 1e-8
            direction = normed[..., :d] / dist
            #  contracted = torch.where(dist > 1, (r+1)-r/dist, dist)/2
            contracted = torch.where(dist > 1, (dist-1)/4+1, dist)/2
            return torch.cat([ contracted * direction, normed[..., d:] ], dim=-1)
        else:
            return normed

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        raise Exception("Not implemented")

    def init_svd_volume(self, res, device):
        pass

    def vector_comp_diffs(self):
        raise Exception("Not implemented")

    def compute_features(self, xyz_sampled):
        raise Exception("Not implemented")

    def compute_densityfeature(self, xyz_sampled):
        raise Exception("Not implemented")

    def compute_appfeature(self, xyz_sampled):
        raise Exception("Not implemented")

    def shrink(self, new_aabb, voxel_size):
        raise Exception("Not implemented")
