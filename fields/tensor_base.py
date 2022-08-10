import torch
from icecream import ic
import numpy as np
import utils

class TensorBase(torch.nn.Module):
    def __init__(self, aabb, grid_size, density_n_comp, appearance_n_comp,
                 app_dim, step_ratio, density_res_multi, contract_space,
                 N_voxel_init, N_voxel_final, upsamp_list):
        super().__init__()
        self.separate_appgrid = True
        self.dtype = torch.half
        self.density_n_comp = [density_n_comp]*3
        self.app_n_comp = [appearance_n_comp]*3
        self.density_res_multi = density_res_multi
        self.app_dim = app_dim
        self.register_buffer('aabb', aabb)
        self.step_ratio = step_ratio
        self.contract_space = contract_space
        self.N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(N_voxel_init), np.log(N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]
        self.upsamp_list = upsamp_list

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        print(1, grid_size)
        grid_size = torch.tensor(utils.N_to_reso(N_voxel_init, self.aabb))
        print(2, grid_size)

        self.update_stepSize(grid_size)

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
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)
        
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

    def check_schedule(self, iter):
        if iter in self.upsamp_list:
            i = self.upsamp_list.find(iter)
            n_voxels = self.N_voxel_list[i]
            reso_cur = utils.N_to_reso(n_voxels, self.aabb)
            # nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio/tensorf.rf.density_res_multi))
            self.upsample_volume_grid(reso_cur)
            return True
        return False

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        raise Exception("Not implemented")

    def init_svd_volume(self, res, device):
        pass

    def vector_comp_diffs(self):
        raise Exception("Not implemented")

    def compute_densityfeature(self, xyz_sampled):
        raise Exception("Not implemented")

    def compute_appfeature(self, xyz_sampled):
        raise Exception("Not implemented")

    def shrink(self, new_aabb, voxel_size):
        raise Exception("Not implemented")
