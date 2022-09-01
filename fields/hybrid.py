from .tensor_base import TensorVoxelBase
import torch
import torch.nn.functional as F
from icecream import ic
from models.grid_sample_Cinf import grid_sample
import random
import math
import tinycudann as tcnn
import numpy as np

# here is original grid sample derivative for testing
# def grid_sample(*args, smoothing, **kwargs):
#     return F.grid_sample(*args, **kwargs)


class HybridRF(TensorVoxelBase):
    def __init__(self, aabb, encoder_conf, smoothing=1.5, *args, **kargs):
        super(HybridRF, self).__init__(aabb, *args, **kargs)

        # num_levels x num_outputs
        # self.interp_mode = 'bilinear'
        self.interp_mode = 'bicubic'
        self.align_corners = True

        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, [int(self.density_res_multi*g) for g in self.grid_size], 0.1, -0)
        self.dbasis_mat = torch.nn.Linear(sum(self.density_n_comp), 1, bias=False)

        self.smoothing = smoothing

        bound = 1
        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))
        self.encoding = tcnn.Encoding(3, encoding_config=dict(per_level_scale=per_level_scale, **encoder_conf))
        self.app_dim = encoder_conf.n_features_per_level * encoder_conf.n_levels

    def init_one_svd(self, n_component, grid_size, scale, shift):
        plane_coef, line_coef = [], []

        xyg = torch.meshgrid(torch.linspace(-1, 1, grid_size[0]), torch.linspace(-1, 1, grid_size[1]), indexing='ij')
        xy = xyg[0] + xyg[1]

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            pos_vals = xy.reshape(1, 1, grid_size[mat_id_0], grid_size[mat_id_1])
            # freqs = torch.arange(n_component[i]//2).reshape(1, -1, 1, 1)
            freqs = 2**torch.arange(n_component[i]//2-1).reshape(1, -1, 1, 1)
            freqs = torch.cat([torch.zeros_like(freqs[:, 0:1]), freqs], dim=1)
            line_pos_vals = torch.linspace(-1, 1, grid_size[vec_id]).reshape(1, 1, -1, 1)
            scales = scale * 1/(freqs+1)
            # scales[:, scales.shape[1]//2:] = 0
            plane_coef_v = torch.nn.Parameter(
                torch.cat([
                    scales * torch.sin(freqs * pos_vals * math.pi),
                    scales * torch.cos(freqs * pos_vals * math.pi),
                ], dim=1)
                # scale * torch.randn((1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0]))
                # scale * torch.rand((1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0])) + shift/sum(n_component)
            )
            line_coef_v = torch.nn.Parameter(
                torch.cat([
                    scales * torch.sin(freqs * line_pos_vals * math.pi),
                    scales * torch.cos(freqs * line_pos_vals * math.pi),
                ], dim=1)
                # scale * torch.randn((1, n_component[i], grid_size[vec_id], 1))
                # scale * torch.rand((1, n_component[i], grid_size[vec_id], 1))
            )
            # adjust parameter so the density is always > 0
            plane_coef.append(plane_coef_v)
            line_coef.append(line_coef_v)

        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)
    
    
    def get_optparam_groups(self):
        grad_vars = [
            {'params': self.density_line, 'lr': self.lr}, {'params': self.density_plane, 'lr': self.lr},
            {'params': self.dbasis_mat.parameters(), 'lr': self.lr_net},
            {'params': self.encoding.parameters(), 'lr': self.lr},
        ]
        return grad_vars

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
        return total

    def coords2input(self, xyz_normed):
        return (xyz_normed[..., :3].reshape(-1, 3)/2+0.5).contiguous()

    def coordinates(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
        return coordinate_plane, coordinate_line

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane, coordinate_line = self.coordinates(xyz_sampled)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        sigma_feature = []

        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=self.align_corners, mode=self.interp_mode, smoothing=self.smoothing).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=self.align_corners, mode=self.interp_mode, smoothing=self.smoothing).view(-1, *xyz_sampled.shape[:1])
            sigma_feature.append(plane_coef_point * line_coef_point)

        # return self.dbasis_mat(sigma_feature.reshape(-1, 1)).reshape(-1)
        sigma_feature = torch.cat(sigma_feature, dim=0).T
        # ic(sigma_feature[0], sigma_feature[0].sum())
        sigma_feature = self.dbasis_mat(sigma_feature).squeeze(-1)
        # sigma_feature = sigma_feature.sum(dim=-1)
        return self.feature2density(sigma_feature)


    def compute_appfeature(self, xyz_normed):
        feat = self.encoding(xyz_normed[..., :3].reshape(-1, 3).contiguous()).type(xyz_normed.dtype)
        return feat


    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode=self.interp_mode,
                              align_corners=self.align_corners))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode=self.interp_mode, align_corners=self.align_corners))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        density_target = [int(self.density_res_multi*g) for g in res_target]
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, density_target)

        self.update_stepSize(res_target)
        print(f'upsampling to {res_target}. upsampling density to {density_target}')

    @torch.no_grad()
    def shrink(self, new_aabb, apply_correction):
        # the new_aabb is in normalized coordinates, from -1 to 1
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
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
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,dt_l[mode1]:db_r[mode1],dt_l[mode0]:db_r[mode0]]
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
        # self.aabb *= new_aabb.abs()
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
