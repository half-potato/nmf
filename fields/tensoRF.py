from .tensor_base import TensorVoxelBase
import torch
import torch.nn.functional as F
from icecream import ic
from models.grid_sample_Cinf import grid_sample
import random
import math
from models import safemath

# here is original grid sample derivative for testing
# def grid_sample(*args, smoothing, **kwargs):
#     return F.grid_sample(*args, **kwargs)

class TensoRF(torch.nn.Module):
    def __init__(self, grid_size, dim, init_mode, interp_mode, init_val, lr, smoothing=0.5):
        super().__init__()

        # tensorf
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.grid_size = grid_size
        self.align_corners = True
        self.lr = lr
        self.smoothing = smoothing
        self.init_mode = init_mode
        self.interp_mode = interp_mode
        self.app_plane, self.app_line = self.init_one_svd(dim, self.grid_size, init_val, 0)

    def get_optparam_groups(self, lr_scale=1):
        grad_vars = [
            {'params': self.app_plane.parameters(), 'lr': self.lr*lr_scale},
            {'params': self.app_line.parameters(), 'lr': self.lr*lr_scale},
        ]
        return grad_vars

    def init_one_svd(self, n_component, grid_size, scale, shift):
        plane_coef, line_coef = [], []


        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            xyg = torch.meshgrid(torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size), indexing='ij')
            xy = xyg[0] + xyg[1]
            pos_vals = xy.reshape(1, 1, grid_size, grid_size)
            # freqs = torch.arange(n_component[i]//2).reshape(1, -1, 1, 1)
            n_degs = n_component//2
            freqs = 2**torch.arange(n_degs-1).reshape(1, -1, 1, 1)
            freqs = torch.cat([torch.zeros_like(freqs[:, 0:1]), freqs], dim=1)
            line_pos_vals = torch.linspace(-1, 1, grid_size).reshape(1, 1, -1, 1)
            scales = scale * 1/(freqs+1)
            # scales[:, scales.shape[1]//2:] = 0
            match self.init_mode:
                case 'trig':
                    plane_coef_v = torch.cat([
                        scales * torch.sin(freqs * pos_vals * math.pi),
                        scales * torch.cos(freqs * pos_vals * math.pi),
                    ], dim=1)
                    line_coef_v = torch.cat([
                        scales * torch.sin(freqs * line_pos_vals * math.pi),
                        scales * torch.cos(freqs * line_pos_vals * math.pi),
                    ], dim=1)
                case 'integrated':
                    b = safemath.integrated_pos_enc((pos_vals.reshape(-1, 1)*torch.pi, torch.zeros_like(pos_vals).reshape(-1, 1)), 0, n_degs)
                    b = b.T.reshape(1, b.shape[1], *pos_vals.shape[-2:])

                    a = safemath.integrated_pos_enc((line_pos_vals.reshape(-1, 1)*torch.pi, torch.zeros_like(line_pos_vals).reshape(-1, 1)), 0, n_degs)
                    a = a.T.reshape(1, a.shape[1], *line_pos_vals.shape[-2:])
                    plane_coef_v = b
                    line_coef_v = a
                case 'randplane':
                    plane_coef_v = scale**(1/2) * torch.randn((1, n_component, grid_size, grid_size))
                    line_coef_v = scale**(1/2) * torch.ones((1, n_component, grid_size, 1))
                case _:
                    plane_coef_v = scale**(1/2) * torch.randn((1, n_component, grid_size, grid_size))
                    line_coef_v = scale**(1/2) * torch.randn((1, n_component, grid_size, 1))
            plane_coef.append(torch.nn.Parameter(plane_coef_v))
            line_coef.append(torch.nn.Parameter(line_coef_v))

        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)

    def coords(self, xyz_normed):
        coordinate_plane = torch.stack((xyz_normed[..., self.matMode[0]], xyz_normed[..., self.matMode[1]], xyz_normed[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_normed[..., self.vecMode[0]], xyz_normed[..., self.vecMode[1]], xyz_normed[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
        return coordinate_plane, coordinate_line

    def forward(self, xyz_normed):
        coordinate_plane, coordinate_line = self.coords(xyz_normed)
        feature = []
        for idx_plane in range(len(self.app_plane)):
            pc = grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]], mode=self.interp_mode,
                        align_corners=self.align_corners, smoothing=self.smoothing).view(-1, *xyz_normed.shape[:1])
            lc = grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=self.align_corners, mode=self.interp_mode, smoothing=self.smoothing).view(-1, *xyz_normed.shape[:1])
            feature.append(pc * lc)
        return feature

    @torch.no_grad()
    def upsample(self, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            self.app_plane[i] = torch.nn.Parameter(
                F.interpolate(self.app_plane[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode=self.interp_mode,
                              align_corners=self.align_corners))
            self.app_line[i] = torch.nn.Parameter(
                F.interpolate(self.app_line[i].data, size=(res_target[vec_id], 1), mode=self.interp_mode, align_corners=self.align_corners))

    @torch.no_grad()
    def shrink(self, t_l, b_r):
        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            # ic(self.density_plane[i].data.shape)
            # ic(self.density_line[i].data.shape)


class Triplanar(TensoRF):

    def forward(self, xyz_normed):
        coordinate_plane, coordinate_line = self.coords(xyz_normed)
        coefs = 1
        for idx_plane in range(len(self.app_plane)):
            pc = grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]], mode=self.interp_mode,
                        align_corners=self.align_corners, smoothing=self.smoothing).view(-1, *xyz_normed.shape[:1])
            coefs = pc * coefs
        return coefs.T


class TensorVMSplit(TensorVoxelBase):
    def __init__(self, aabb, smoothing, interp_mode = 'bilinear', dbasis=True, init_mode='trig', *args, **kwargs):
        super(TensorVMSplit, self).__init__(aabb, *args, **kwargs)

        # num_levels x num_outputs
        self.interp_mode = interp_mode
        self.init_mode = init_mode
        self.dbasis = dbasis
        # self.interp_mode = 'bicubic'
        self.align_corners = True

        #(grid_size, dim, init_mode, interp_mode, init_val, smoothing=0.5):
        self.density_rf = TensoRF(self.grid_size[0], self.density_n_comp[0], init_mode, interp_mode, 0.1, self.lr, smoothing)
        self.app_rf = TensoRF(self.grid_size[0], self.app_n_comp[0], init_mode, interp_mode, 0.1, self.lr, smoothing)
        m = sum(self.app_n_comp)
        self.basis_mat = torch.nn.Linear(m, self.app_dim, bias=False)
        self.dbasis_mat = torch.nn.Linear(sum(self.density_n_comp), 1, bias=False)

        self.smoothing = smoothing
    
    
    def get_optparam_groups(self, lr_scale):
        grad_vars = [
            {'params': self.basis_mat.parameters(), 'lr': lr_scale * self.lr_net, 'betas': [0.9, 0.999]},
            {'params': self.dbasis_mat.parameters(), 'lr': lr_scale * self.lr_net, 'betas': [0.9, 0.999]},
            *self.density_rf.get_optparam_groups(lr_scale),
            *self.app_rf.get_optparam_groups(lr_scale),
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
        
    def TV_loss_app(self, reg, start_ind=0, end_ind=-1):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
        return total

    def coordinates(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
        return coordinate_plane, coordinate_line

    def _compute_densityfeature(self, xyz_sampled):
        sigma_feature = self.density_rf(xyz_sampled)

        if self.dbasis:
            sigma_feature = torch.cat(sigma_feature, dim=0).T
            sigma_feature = self.dbasis_mat(sigma_feature).squeeze(-1)
        else:
            sigma_feature = sum(sigma_feature).sum(dim=0)
        return sigma_feature

    def _compute_appfeature(self, xyz_sampled):
        coefs = self.app_rf(xyz_sampled)
        coefs = torch.cat(coefs, dim=0).T
        return self.basis_mat(coefs)


    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_rf.upsample(res_target)
        self.density_rf.upsample(res_target)

        self.update_stepSize(res_target)
        print(f'upsampling to {res_target}. upsampling density to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb, mask_gridsize):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        ic(t_l, b_r)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)
        t_l = t_l.clip(min=0)
        ic(t_l, b_r)

        # if not torch.all(mask_gridsize == self.grid_size):
        t_l_r, b_r_r = t_l / (self.grid_size-1), (b_r-1) / (self.grid_size-1)
        correct_aabb = torch.zeros_like(new_aabb)
        correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
        correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
        print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
        new_aabb = correct_aabb
        if torch.equal(new_aabb, self.aabb):
            return

        self.app_rf.shrink(t_l, b_r)
        self.density_rf.shrink(t_l, b_r)

        newSize = b_r - t_l
        ic(newSize)
        self.set_aabb(new_aabb)
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
