from tkinter import W
from .tensor_base import TensorBase
import torch
import torch.nn.functional as F
from icecream import ic
import time

class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, *args, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize, device, *args, **kargs)
        

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device))
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device))
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False, device=device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):

        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach()
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach()

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return sigma_feature, app_features

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        
        return app_features
    

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            # print(self.line_coef.shape, vector_comps[idx].shape)
            n_comp, n_size = vector_comps[idx].shape[:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape)
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape,non_diagonal.shape)
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.line_coef[:,-self.density_n_comp:]) + self.vectorDiffs(self.line_coef[:,:self.app_n_comp])
    
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        scale = res_target[0]/self.line_coef.shape[2] #assuming xyz have the same scale
        plane_coef = F.interpolate(self.plane_coef.detach().data, scale_factor=scale, mode='bilinear',align_corners=True)
        line_coef  = F.interpolate(self.line_coef.detach().data, size=(res_target[0],1), mode='bilinear',align_corners=True)
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.compute_stepSize(res_target)
        print(f'upsamping to {res_target}')


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, *args, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, *args, **kargs)
        # self.f_blur = torch.tensor([1, 2, 1], device=device) / 4
        f_blur = torch.tensor([0, 1, 0], device=device)
        f_edge = torch.tensor([-1, 0, 1], device=device) / 2
        # f_blur = torch.tensor([1, 1], device=device) / 2
        # f_edge = torch.tensor([-1, 1], device=device) / 2
        l = len(f_blur)

        self.dy_filter = (f_blur[None, :] * f_edge[:, None]).reshape(1, 1, l, l)#.expand(1, self.density_n_comp[0], 3, 3)
        self.dx_filter = self.dy_filter.permute(0, 1, 3, 2)
        self.dz_filter = f_edge.reshape(1, 1, l)#.expand(1, self.density_n_comp[0], 3)
        # random_offset = torch.rand(1, self.density_n_comp[0], 3)
        # self.register_buffer('random_offset', random_offset)

        self.sizes = [2*n+1 for n in range(1, 3)]
        self.sizes = []
        self.plane_kernels = [None, *[torch.ones((n, n), dtype=torch.float32, device=device) for n in self.sizes]]
        self.line_kernels = [None, *[torch.ones((n), dtype=torch.float32, device=device) for n in self.sizes]]
        self.sizes = [1, *self.sizes]


    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
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

    def multi_size_plane(self, plane, coordinate_plane, size_weights, convs=[None]):
        # plane.shape: 1, n_comp, gridSize, gridSize
        # coordinate_plane.shape: 1, N, 1, 2
        # conv.shape: 1, 1, s, s
        
        n_comp = plane.shape[1]

        p_plane = plane.permute(1, 0, 2, 3)

        out = []
        for i, kernel in enumerate(self.plane_kernels):
            if kernel is None:
                level = p_plane
            else:
                s = kernel.shape[-1]
                level = F.conv2d(p_plane, kernel.reshape(1, 1, s, s), stride=1, padding=s//2)
            # level.shape: n_comp, 1, gridSize, gridSize
            for conv in convs:
                if conv is not None:
                    clevel = F.conv2d(clevel, conv, stride=1, padding=conv.shape[-1]//2)
                else:
                    clevel = level
                # clevel.shape: n_comp, 1, gridSize, gridSize
                out.append(clevel.permute(1, 0, 2, 3))
        # conv1*size1, conv2*size1, conv1*size2, conv2*size2

        out = torch.cat(out, dim=1)
        
        ms_plane_coef = F.grid_sample(out, coordinate_plane, mode='bicubic', align_corners=True)
        # ms_line_coef.shape: len(self.line_kernels), len(convs), n_comp, N
        ms_plane_coef = ms_plane_coef.reshape(len(self.plane_kernels), len(convs), n_comp, -1).permute(0, 1, 2, 3)
        # line_coef.shape: len(convs), n_comp, N
        plane_coef = (ms_plane_coef * size_weights.reshape(len(self.plane_kernels), 1, 1, -1)).sum(dim=0)
        output = [plane_coef[i] for i in range(len(convs))]
        if len(output) == 1:
            return output[0]
        return output

    def multi_size_line(self, line, coordinate_line, size_weights, convs=[None]):
        # plane.shape: 1, n_comp, gridSize, 1
        # coordinate_plane.shape: 1, N, 1, 2
        # conv.shape: 1, 1, s
        
        n_comp = line.shape[1]

        p_line = line.permute(1, 0, 2, 3).squeeze(-1)

        out = []
        for i, kernel in enumerate(self.line_kernels):
            if kernel is None:
                level = p_line
            else:
                s = kernel.shape[-1]
                level = F.conv1d(p_line, kernel.reshape(1, 1, s), stride=1, padding=s//2)

            # level.shape: n_comp, 1, gridSize
            for conv in convs:
                if conv is not None:
                    clevel = F.conv1d(level, conv, stride=1, padding=conv.shape[-1]//2)
                else:
                    clevel = level
                # clevel.shape: n_comp, 1, gridSize

                clevel = clevel.unsqueeze(-1).permute(1, 0, 2, 3)
                # clevel.shape: 1, n_comp, gridSize, 1
                out.append(clevel)
        out = torch.cat(out, dim=1)
        
        ms_line_coef = F.grid_sample(out, coordinate_line, mode='bicubic', align_corners=True)
        # ms_line_coef.shape: len(self.line_kernels), len(convs), n_comp, N
        ms_line_coef = ms_line_coef.reshape(len(self.line_kernels), len(convs), n_comp, -1).permute(0, 1, 2, 3)
        # line_coef.shape: len(convs), n_comp, N
        line_coef = (ms_line_coef * size_weights.view(len(self.line_kernels), 1, 1, -1)).sum(dim=0)
        # deliberately split the result to avoid confusion
        output = [line_coef[i] for i in range(len(convs))]
        if len(output) == 1:
            return output[0]
        return output

    def coordinates(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        return coordinate_plane, coordinate_line

    def compute_size_weights(self, xyz_sampled):
        size = xyz_sampled[..., 3]
        # the sum across the weights is 1
        # just want the closest grid point
        # first calculate the size of voxels in meters
        voxel_sizes = self.sizes
        # voxel_sizes = [level.units.max() for level in self.levels]
        voxel_sizes = torch.tensor(voxel_sizes, dtype=torch.float32, device=xyz_sampled.device)

        # then, the weight should be summed from the smallest supported size to the largest
        size_diff = abs(voxel_sizes.reshape(1, -1) - size.reshape(-1, 1))/voxel_sizes.max()
        weights = F.softmax(-size_diff.reshape(*size.shape, len(voxel_sizes)), dim=-1)
        return weights

    # def compute_densityfeature(self, xyz_sampled):
    #     coordinate_plane, coordinate_line = self.coordinates(xyz_sampled)
    #     sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
    #     size_weights = self.compute_size_weights(xyz_sampled)
    #     for idx_plane in range(len(self.density_plane)):
    #         plane_coef_point = self.multi_size_plane(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], size_weights)
    #         line_coef_point = self.multi_size_line(self.density_line[idx_plane], coordinate_line[[idx_plane]], size_weights)
    #         sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
    #     return sigma_feature

    # def compute_density_norm(self, xyz_sampled):
    #     coordinate_plane, coordinate_line = self.coordinates(xyz_sampled)
    #     sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
    #     world_normals = torch.zeros((xyz_sampled.shape[0], 3), device=xyz_sampled.device)
    #     size_weights = self.compute_size_weights(xyz_sampled)
    #     for idx_plane in range(len(self.density_plane)):
    #         plane_coef_point, dx_point, dy_point = self.multi_size_plane(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], size_weights, convs=[None, self.dx_filter, self.dy_filter])
    #         line_coef_point, dz_point = self.multi_size_line(self.density_line[idx_plane], coordinate_line[[idx_plane]], size_weights, convs=[None, self.dz_filter])
    #         sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

    #         # world_normals[:, self.matMode[idx_plane][0]] += (F.relu(line_coef_point)*dx_point).sum(dim=0)
    #         # world_normals[:, self.matMode[idx_plane][1]] += (F.relu(line_coef_point)*dy_point).sum(dim=0)
    #         # world_normals[:, self.vecMode[idx_plane]] += (F.relu(plane_coef_point)*dz_point).sum(dim=0)
    #     return sigma_feature, world_normals

    # def compute_appfeature(self, xyz_sampled):
    #     coordinate_plane, coordinate_line = self.coordinates(xyz_sampled)
    #     size_weights = self.compute_size_weights(xyz_sampled)
    #     plane_coef_point,line_coef_point = [],[]
    #     for idx_plane in range(len(self.app_plane)):
    #         plane_coef_point.append(self.multi_size_plane(self.app_plane[idx_plane], coordinate_plane[[idx_plane]], size_weights))
    #         line_coef_point.append(self.multi_size_line(self.app_line[idx_plane], coordinate_line[[idx_plane]], size_weights))
    #     plane_coef_point, line_coef_point = torch.cat(plane_coef_point, dim=0), torch.cat(line_coef_point, dim=0)
    #     return self.basis_mat((plane_coef_point * line_coef_point).T)

# """
    def compute_densityfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane, coordinate_line = self.coordinates(xyz_sampled)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                mode='bicubic', align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            mode='bicubic', align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
        return sigma_feature

    def compute_density_norm(self, xyz_sampled):

        # plane + line basis
        normal_feature = torch.zeros((xyz_sampled.shape[0], 3), device=xyz_sampled.device)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)

        coordinate_plane, coordinate_line = self.coordinates(xyz_sampled)
        # coordinate_plane.shape: torch.Size([3, 1050944, 1, 2])
        # xyz_sampled.shape: torch.Size([1050944, 4])

        for idx_plane in range(len(self.density_plane)):
            # print(idx_plane, self.density_plane[idx_plane].shape)
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                mode='bicubic', align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            mode='bicubic', align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

            plane = (self.density_plane[idx_plane]).permute(1, 0, 2, 3)
            line = (self.density_line[idx_plane]).permute(1, 0, 2, 3)
            dx_plane = F.conv2d(plane, self.dx_filter, stride=1, padding=1).permute(1, 0, 2, 3)
            dy_plane = F.conv2d(plane, self.dy_filter, stride=1, padding=1).permute(1, 0, 2, 3)
            dz_line = F.conv1d(line.squeeze(-1), self.dz_filter, stride=1, padding=1).unsqueeze(-1).permute(1, 0, 2, 3)

            dx_point = F.grid_sample(dx_plane, coordinate_plane[[idx_plane]],
                                     mode='bicubic', align_corners=True).view(-1, *xyz_sampled.shape[:1])
            dy_point = F.grid_sample(dy_plane, coordinate_plane[[idx_plane]],
                                     mode='bicubic', align_corners=True).view(-1, *xyz_sampled.shape[:1])
            dz_point = F.grid_sample(dz_line, coordinate_line[[idx_plane]],
                                     mode='bicubic', align_corners=True).view(-1, *xyz_sampled.shape[:1])
            # orient based on mat mode
            normal_feature[:, self.matMode[idx_plane][0]] += (F.relu(line_coef_point)*dx_point).sum(dim=0)
            normal_feature[:, self.matMode[idx_plane][1]] += (F.relu(line_coef_point)*dy_point).sum(dim=0)
            normal_feature[:, self.vecMode[idx_plane]] += (F.relu(plane_coef_point)*dz_point).sum(dim=0)

        normal_feature = normal_feature / (torch.norm(normal_feature, dim=1, keepdim=True)+1e-6)

        return sigma_feature, normal_feature

    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane, coordinate_line = self.coordinates(xyz_sampled)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)



    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb, apply_correction):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        # if not torch.all(self.alphaMask.gridSize == self.gridSize):
        if apply_correction:
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, *args, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, device, *args, **kargs)


    def init_svd_volume(self, res, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2, device)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.app_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)
    

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)


        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total