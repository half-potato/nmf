from .tensor_base import TensorBase
import torch
import torch.nn.functional as F
from icecream import ic
from .convolver import Convolver
from .grid_sample_Cinf import grid_sample
from .grid_sample3d import grid_sample_3d
import random
import math

class Grid(TensorBase):
    def __init__(self, aabb, grid_size, *args, **kargs):
        super(Grid, self).__init__(aabb, grid_size, *args, **kargs)

        # num_levels x num_outputs
        self.interp_mode = 'bilinear'
        self.align_corners = True

        density_grid = torch.rand(1, 1, *grid_size)
        app_grid = torch.rand(1, self.app_dim, *grid_size)
        self.register_parameter('density_grid', torch.nn.Parameter(density_grid))
        self.register_parameter('app_grid', torch.nn.Parameter(app_grid))


    def set_smoothing(self, sm):
        pass

    def init_one_svd(self, n_component, grid_size, scale, shift):
        pass
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [
            {'params': self.density_grid, 'lr': lr_init_spatialxyz},
            {'params': self.app_grid, 'lr': lr_init_spatialxyz},
        ]
        return grad_vars

    def vector_comp_diffs(self):
        return
    
    def density_L1(self):
        return torch.abs(self.density_grid).mean()
    
    def TV_loss_density(self, reg):
        return reg(self.density_grid)

    def TV_loss_app(self, reg):
        return reg(self.app_grid)

    def compute_densityfeature(self, xyz_sampled):
        # a = F.grid_sample(self.density_grid, xyz_sampled.reshape(1, 1, 1, -1, xyz_sampled.shape[-1])[..., :3], mode=self.interp_mode,
        #     align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1])
        # return a.reshape(-1)
        a = grid_sample_3d(
                self.density_grid,
                xyz_sampled.reshape(1, 1, 1, -1, xyz_sampled.shape[-1])[..., :3]).view(-1, *xyz_sampled.shape[:1])
        return a.reshape(-1)

    def compute_appfeature(self, xyz_sampled):
        a = F.grid_sample(self.app_grid, xyz_sampled.reshape(1, 1, 1, -1, xyz_sampled.shape[-1])[..., :3], mode=self.interp_mode,
            align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1])
        return a.view(-1, *xyz_sampled.shape[:1]).T

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        density_target = [int(self.density_res_multi*g) for g in res_target]
        self.app_grid = torch.nn.Parameter(
            F.interpolate(self.app_grid.data, size=res_target, mode='trilinear',
                          align_corners=self.align_corners))
        self.density_grid = torch.nn.Parameter(
            F.interpolate(self.density_grid.data, size=density_target, mode='trilinear',
                          align_corners=self.align_corners))

        self.update_stepSize(res_target)
        print(f'upsampling to {res_target}. upsampling density to {density_target}')

    @torch.no_grad()
    def shrink(self, new_aabb, apply_correction):
        return
        # the new_aabb is in normalized coordinates, from -1 to 1
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb * self.aabb.abs()
        # t_l, b_r = xyz_min * self.grid_size // 2, xyz_max * self.grid_size // 2 - 1
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        dt_l, db_r = torch.round(t_l*self.density_res_multi).long(), torch.round(b_r*self.density_res_multi).long() + 1
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)
        db_r = torch.stack([db_r, (self.density_res_multi*self.grid_size).long()]).amin(0)
        ic(db_r, dt_l, b_r, t_l, xyz_min, xyz_max, self.units, self.aabb, self.density_line[0].shape, self.grid_size)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,dt_l[mode0]:db_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,dt_l[mode1]:db_r[mode1],dt_l[mode0]:db_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        # if not torch.all(self.alphaMask.grid_size == self.grid_size):
        if apply_correction:
            t_l_r, b_r_r = t_l / (self.grid_size-1), (b_r-1) / (self.grid_size-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb *= new_aabb.abs()
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

