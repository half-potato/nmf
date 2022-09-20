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

def d_softplus(x, beta=1.0, shift=-10):
    return torch.exp(shift+beta*x) / (1.0 + torch.exp(shift+beta*x))


class Triplanar(TensorVoxelBase):
    def __init__(self, aabb, init_mode='trig', *args, smoothing, **kargs):
        super(Triplanar, self).__init__(aabb, *args, **kargs)

        # num_levels x num_outputs
        # self.interp_mode = 'bilinear'
        self.init_mode = init_mode
        self.interp_mode = 'bilinear'
        self.align_corners = True

        self.density_plane = self.init_one_svd(self.density_n_comp, [int(self.density_res_multi*g) for g in self.grid_size], 0.1, -0)
        self.app_plane = self.init_one_svd(self.app_n_comp, self.grid_size, 0.1, 0)
        m = sum(self.app_n_comp)
        self.basis_mat = torch.nn.Linear(m, self.app_dim, bias=False)
        self.dbasis_mat = torch.nn.Linear(sum(self.density_n_comp), 1, bias=False)

        self.smoothing = smoothing

    def init_one_svd(self, n_component, grid_size, scale, shift):
        plane_coef = []

        xyg = torch.meshgrid(torch.linspace(-1, 1, grid_size[0]), torch.linspace(-1, 1, grid_size[1]), indexing='ij')
        xy = xyg[0] + xyg[1]

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            pos_vals = xy.reshape(1, 1, grid_size[mat_id_0], grid_size[mat_id_1])
            # freqs = torch.arange(n_component[i]//2).reshape(1, -1, 1, 1)
            n_degs = n_component[i]//2
            freqs = 2**torch.arange(n_degs-1).reshape(1, -1, 1, 1)
            freqs = torch.cat([torch.zeros_like(freqs[:, 0:1]), freqs], dim=1)
            line_pos_vals = torch.linspace(-1, 1, grid_size[vec_id]).reshape(1, 1, -1, 1)
            scales = scale * 1/(freqs+1)
            # scales[:, scales.shape[1]//2:] = 0
            match self.init_mode:
                case 'trig':
                    plane_coef_v = torch.cat([
                        scales * torch.sin(freqs * pos_vals * math.pi),
                        scales * torch.cos(freqs * pos_vals * math.pi),
                    ], dim=1)
                case 'integrated':
                    b = safemath.integrated_pos_enc((pos_vals.reshape(-1, 1)*torch.pi, torch.ones_like(pos_vals).reshape(-1, 1)), 0, n_degs)
                    b = b.T.reshape(1, b.shape[1], *pos_vals.shape[-2:])

                    a = safemath.integrated_pos_enc((line_pos_vals.reshape(-1, 1)*torch.pi, torch.ones_like(line_pos_vals).reshape(-1, 1)), 0, n_degs)
                    a = a.T.reshape(1, a.shape[1], *line_pos_vals.shape[-2:])
                    plane_coef_v = b
                case 'rand':
                    plane_coef_v = scale * torch.randn((1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0]))
            plane_coef.append(torch.nn.Parameter(plane_coef_v))

        return torch.nn.ParameterList(plane_coef)
    
    
    def get_optparam_groups(self):
        grad_vars = [
            {'params': self.density_plane, 'lr': self.lr}, 
            {'params': self.app_plane, 'lr': self.lr},
            {'params': self.basis_mat.parameters(), 'lr': self.lr_net},
            {'params': self.dbasis_mat.parameters(), 'lr': self.lr_net},
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
            total = total + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2
        return total
        
    def TV_loss_app(self, reg, start_ind=0, end_ind=-1):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2
        return total

    def coordinates(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        return coordinate_plane

    def compute_densityfeature(self, xyz_sampled):
        # mask1 = torch.linalg.norm(xyz_sampled[..., :3], dim=-1, ord=torch.inf) < 0.613/1.5
        # mask2 = (xyz_sampled[..., 0] < 0) & (xyz_sampled[..., 1] > 0)
        # return torch.where(mask1 & ~mask2, 99999999.0, 0.0)

        coordinate_plane = self.coordinates(xyz_sampled)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        sigma_feature = []

        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=self.align_corners, mode=self.interp_mode, smoothing=self.smoothing).view(-1, *xyz_sampled.shape[:1])
            sigma_feature.append(plane_coef_point)

        # return self.dbasis_mat(sigma_feature.reshape(-1, 1)).reshape(-1)
        sigma_feature = torch.cat(sigma_feature, dim=0).T
        # ic(sigma_feature[0], sigma_feature[0].sum())
        sigma_feature = self.dbasis_mat(sigma_feature).squeeze(-1)
        # ic(list(self.dbasis_mat.parameters()))
        # sigma_feature = (sigma_feature).sum(dim=1).squeeze(-1)
        # sigma_feature = sigma_feature.sum(dim=-1)
        return self.feature2density(sigma_feature)


    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = self.coordinates(xyz_sampled)
        plane_coef_point = []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(
                    F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]], mode=self.interp_mode,
                        align_corners=self.align_corners).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point = torch.cat(plane_coef_point, dim=0)
        return self.basis_mat(plane_coef_point.T)


    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode=self.interp_mode,
                              align_corners=self.align_corners))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        density_target = [int(self.density_res_multi*g) for g in res_target]
        self.app_plane = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane = self.up_sampling_VM(self.density_plane, self.density_line, density_target)

        self.update_stepSize(res_target)
        print(f'upsampling to {res_target}. upsampling density to {density_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        # the new_aabb is in normalized coordinates, from -1 to 1
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        # t_l, b_r = xyz_min * self.grid_size // 2, xyz_max * self.grid_size // 2 - 1
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        dt_l, db_r = torch.floor(t_l*self.density_res_multi).long(), torch.ceil(b_r*self.density_res_multi).long() + 1
        t_l, b_r = torch.floor(t_l).long(), torch.ceil(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)
        db_r = torch.stack([db_r, (self.density_res_multi*self.grid_size).long()]).amin(0)

        # update aabb
        l1 = t_l / self.grid_size
        l2 = b_r / self.grid_size
        adj_aabb = torch.stack([
            l1 * self.aabb[1] + (1-l1) * self.aabb[0],
            l2 * self.aabb[1] + (1-l2) * self.aabb[0],
        ], dim=0)
        ic(db_r, dt_l, b_r, t_l, xyz_min, xyz_max, self.units, self.aabb, adj_aabb, self.density_line[0].shape, self.grid_size)
        self.aabb = adj_aabb

        for i in range(len(self.vecMode)):
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,dt_l[mode1]:db_r[mode1],dt_l[mode0]:db_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        newSize = b_r - t_l
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
