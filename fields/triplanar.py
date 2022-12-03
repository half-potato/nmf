from tkinter import W
from .tensor_base import TensorVoxelBase
import torch
import torch.nn.functional as F
from icecream import ic
import time
from .convolver import Convolver

class Triplanar(TensorVoxelBase):
    def __init__(self, aabb, grid_size, *args, hier_sizes, **kargs):
        super(Triplanar, self).__init__(aabb, grid_size, *args, **kargs)


        self.convolver = Convolver(hier_sizes, False)
        self.sizes = self.convolver.sizes

        # num_levels x num_outputs
        self.interp_mode = 'bilinear'
        # self.interp_mode = 'bicubic'
        self.set_smoothing(1.0)

    def set_smoothing(self, sm):
        self.convolver.set_smoothing(sm)

    def init_svd_volume(self, res):
        self.density_plane = self.init_one_svd(self.density_n_comp, [int(self.density_res_multi*g) for g in self.grid_size], 0.1, -0)
        self.app_plane = self.init_one_svd(self.app_n_comp, self.grid_size, 0.1, 0)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False)


    def init_one_svd(self, n_component, grid_size, scale, shift):
        plane_coef = []
        for i in range(len(self.vecMode)):
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0])) + shift/sum(n_component)
            ))

        return torch.nn.ParameterList(plane_coef)
    
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
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
        return self.vectorDiffs(self.density_line)
    
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
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2
        return total

    def coordinates(self, xyz_sampled):
        # coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        # coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        # coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        return coordinate_plane


    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = self.coordinates(xyz_sampled)
        sigma_feature = torch.zeros((self.density_n_comp[0], xyz_sampled.shape[0]), device=xyz_sampled.device)
        size_weights = self.convolver.compute_size_weights(xyz_sampled, self.units/self.density_res_multi)
        # plane_kerns, line_kerns = self.convolver.get_kernels(size_weights)
        plane_kerns, line_kerns = [[None]], [[None]]

        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = self.convolver.multi_size_plane(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], size_weights, plane_kerns)
            # ic(plane_coef_point.mean(), line_coef_point.mean())
            sigma_feature = sigma_feature * plane_coef_point
        return sigma_feature.sum(dim=0)

    def compute_density_norm(self, xyz_sampled, activation_fn):
        coordinate_plane = self.coordinates(xyz_sampled)
        world_normals = torch.zeros((xyz_sampled.shape[0], 3), device=xyz_sampled.device)
        size_weights = self.convolver.compute_size_weights(xyz_sampled, self.units/self.density_res_multi)

        plane_kerns, line_kerns = self.convolver.get_kernels(size_weights, with_normals=True)

        # self.matMode = [[0,1], [0,2], [1,2]]
        plane_coef_point1, dx_point1, dy_point1 = self.convolver.multi_size_plane(self.density_plane[0], coordinate_plane[[0]], size_weights, convs=plane_kerns)
        plane_coef_point2, dx_point2, dz_point2 = self.convolver.multi_size_plane(self.density_plane[1], coordinate_plane[[1]], size_weights, convs=plane_kerns)
        plane_coef_point3, dy_point3, dz_point3 = self.convolver.multi_size_plane(self.density_plane[2], coordinate_plane[[2]], size_weights, convs=plane_kerns)
        world_normals[:, 0] = (plane_coef_point3*plane_coef_point2*dx_point1 + plane_coef_point1*plane_coef_point3*dx_point2).sum(dim=0)
        world_normals[:, 1] = (plane_coef_point3*plane_coef_point2*dy_point1 + plane_coef_point1*plane_coef_point2*dy_point3).sum(dim=0)
        world_normals[:, 2] = (plane_coef_point3*plane_coef_point1*dz_point2 + plane_coef_point1*plane_coef_point2*dz_point3).sum(dim=0)
        
        sigma_feature = (plane_coef_point1 * plane_coef_point2 * plane_coef_point3).sum(dim=0)

        world_normals = world_normals / (torch.norm(world_normals, dim=1, keepdim=True)+1e-6)
        return sigma_feature, world_normals

    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = self.coordinates(xyz_sampled)
        size_weights = self.convolver.compute_size_weights(xyz_sampled, self.units/self.density_res_multi)
        # app_feature = torch.zeros((self.app_n_comp[0], xyz_sampled.shape[0]), device=xyz_sampled.device)
        app_feature = []
        # plane_kerns = [self.norm_plane_kernels[0][0:1]]
        # line_kerns = [self.norm_line_kernels[0][0:1]]
        plane_kerns, line_kerns = [[None]], [[None]]
        for idx_plane in range(len(self.app_plane)):
            # app_feature *= self.convolver.multi_size_plane(self.app_plane[idx_plane], coordinate_plane[[idx_plane]], size_weights, convs=plane_kerns)
            app_feature.append(self.convolver.multi_size_plane(self.app_plane[idx_plane], coordinate_plane[[idx_plane]], size_weights, convs=plane_kerns))
        app_feature = torch.cat(app_feature, dim=0)
        return self.basis_mat(app_feature.T)


    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))

        return plane_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        density_target = [int(self.density_res_multi*g) for g in res_target]
        self.app_plane = self.up_sampling_VM(self.app_plane, res_target)
        self.density_plane = self.up_sampling_VM(self.density_plane, density_target)

        self.update_stepSize(res_target)
        print(f'upsampling to {res_target}. upsampling density to {density_target}')

    @torch.no_grad()
    def shrink(self, new_aabb, apply_correction):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        dt_l, db_r = torch.round(t_l*self.density_res_multi).long(), torch.round(b_r*self.density_res_multi).long() + 1
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)
        db_r = torch.stack([db_r, (self.density_res_multi*self.grid_size).long()]).amin(0)
        ic(db_r, dt_l, b_r, t_l, xyz_min, xyz_max, self.units, self.aabb)

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
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
