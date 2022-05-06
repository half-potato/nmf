import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time
from icecream import ic

from . import render_modules
import plotly.express as px
import plotly.graph_objects as go

from .tensoRF import TensorCP, TensorVM, TensorVMSplit
from .multi_level_rf import MultiLevelRF

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([
        torch.ones(alpha.shape[0], 1, device=alpha.device),
        1. - alpha + 1e-10
    ], dim=-1), dim=-1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled[..., :3].view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        coords = (xyz_sampled[..., :3]-self.aabb[0]) * self.invgridSize - 1
        size = xyz_sampled[..., 3:4]
        return torch.cat((coords, size), dim=-1)


class TensorNeRF(torch.nn.Module):
    def __init__(self, model_name, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0], enable_reflections = True,
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus', bundle_size = 3, density_grid_dims=8, density_res_multi=1):
        super(TensorNeRF, self).__init__()
        self.rf = eval(model_name)(aabb, gridSize, device, density_n_comp,
                                   appearance_n_comp, app_dim, step_ratio, density_res_multi=density_res_multi, num_levels=3)

        self.model_name = model_name
        self.app_dim = app_dim
        self.alphaMask = alphaMask
        self.device=device
        self.enable_reflections = enable_reflections

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.bundle_size = bundle_size
        self.density_grid_dims = density_grid_dims

        self.f_blur = torch.tensor([1, 2, 1], device=device) / 4
        self.f_edge = torch.tensor([-1, 0, 1], device=device) / 2

        self.max_normal_similarity = np.cos(np.deg2rad(45))
        # self.max_normal_similarity = 1

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        self.has_grid = False
        if shadingMode == 'MLP_PE':
            self.renderModule = render_modules.MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            ref_pe = 6
            self.renderModule = render_modules.MLPRender_Fea(self.app_dim, view_pe, fea_pe, ref_pe, featureC).to(device)
            self.reflectionModule = render_modules.MLPRender_Fea(self.app_dim, view_pe, fea_pe, ref_pe, featureC).to(device)
        elif shadingMode == 'BundleMLP_Fea':
            self.renderModule = render_modules.BundleMLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = render_modules.MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'BundleMLP':
            self.renderModule = render_modules.BundleMLPRender(self.app_dim, view_pe, featureC, self.bundle_size).to(device)
        elif shadingMode == 'BundleMLP_Fea_Grid':
            self.renderModule = render_modules.BundleMLPRender_Fea_Grid(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size, extra=self.density_grid_dims).to(device)
            self.reflectionModule = render_modules.BundleMLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size).to(device)
            self.has_grid = True
        elif shadingMode == 'BundleMLPSphEncGrid':
            self.renderModule = render_modules.BundleMLPRender_Fea_Grid(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size, extra=self.density_grid_dims).to(device)
            # self.reflectionModule = render_modules.BundleSphEncoding(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size).to(device)
            self.reflectionModule = render_modules.BundleMLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size).to(device)
            # self.reflectionModule = render_modules.BundleDirectSphEncoding(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size).to(device)
            self.has_grid = True
        elif shadingMode == 'SH':
            self.renderModule = render_modules.SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = render_modules.RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe, "density_grid_dims", self.density_grid_dims)
        print(self.renderModule)
    
    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        grad_vars = []
        grad_vars += self.rf.get_optparam_groups(lr_init_spatial, lr_init_network)
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        if hasattr(self, 'reflectionModule') and isinstance(self.reflectionModule, torch.nn.Module):
            grad_vars += [{'params':self.reflectionModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def get_kwargs(self):
        return {
            **self.rf.get_kwargs(),
            'model_name': self.model_name,
            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,
            'bundle_size': self.bundle_size,
            'enable_reflections': self.enable_reflections,
            'density_grid_dims': self.density_grid_dims,
            'enable_reflections': self.enable_reflections,
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])


    def compute_normal(self, depth_map, focal):
        # Compute normal map
        dy = (depth_map * (self.f_blur[None, :] * self.f_edge[:, None]).reshape(1, 3, 3)).sum(1).sum(1)
        dx = (depth_map * (self.f_blur[:, None] * self.f_edge[None, :]).reshape(1, 3, 3)).sum(1).sum(1)

        dx = dx * focal * 2 / depth_map[..., self.bundle_size//2, self.bundle_size//2]
        dy = dy * focal * 2 / depth_map[..., self.bundle_size//2, self.bundle_size//2]
        inv_denom = 1 / torch.sqrt(1 + dx**2 + dy**2)
        # (N, 3)
        normal_map = torch.stack([dx * inv_denom, -dy * inv_denom, inv_denom], -1)
        return normal_map

    def sample_ray_ndc(self, rays_o, rays_d, focal, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.rf.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            l = torch.rand_like(interpx)
            interpx += l.to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.rf.aabb[0] > rays_pts) | (rays_pts > self.rf.aabb[1])).any(dim=-1)

        # add size
        rays_pts = torch.cat([rays_pts, interpx.unsqueeze(-1)/focal], dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, focal, is_train=True, N_samples=-1):
        # focal: ratio of meters to pixels at a distance of 1 meter
        N_samples = N_samples if N_samples>0 else self.rf.nSamples
        stepsize = self.rf.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.rf.aabb[1].to(rays_o) - rays_o) / vec
        rate_b = (self.rf.aabb[0].to(rays_o) - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples, device=rays_o.device)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            # N, N_samples
            # add noise along each ray
            brng = rng.reshape(-1, N_samples)
            # brng = brng + torch.rand_like(brng[:, [0], [0]])
            # r = torch.rand_like(brng[:, 0:1, 0:1])
            r = torch.rand_like(brng[:, 0:1])
            brng = brng + r
            rng = brng.reshape(-1, N_samples)
        step = stepsize * rng
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.rf.aabb[0]>rays_pts) | (rays_pts>self.rf.aabb[1])).any(dim=-1)

        # add size
        rays_pts = torch.cat([rays_pts, interpx.unsqueeze(-1)/focal], dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        dense_xyz = torch.stack([*torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2])), 
            torch.ones((gridSize[0], gridSize[1], gridSize[2]))*self.rf.units.min().cpu()*0.5
        ], -1).to(self.device)

        dense_xyz[..., :3] = self.rf.aabb[0] * (1-dense_xyz[..., :3]) + self.rf.aabb[1] * dense_xyz[..., :3]

        # dense_xyz = dense_xyz
        # print(self.rf.stepSize, self.distance_scale*self.rf.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,4), self.rf.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        gridSize = [int(self.rf.density_res_multi*g) for g in gridSize]
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.rf.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.0]

        xyz_min = valid_xyz.amin(0)[:3]
        xyz_max = valid_xyz.amax(0)[:3]

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, focal, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.rf.aabb[1].to(rays_o) - rays_o) / vec
                rate_b = (self.rf.aabb[0].to(rays_o) - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, focal, N_samples=N_samples, is_train=False)
                # Issue: calculate size
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered], mask_filtered


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus_shift":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "softplus":
            return F.softplus(density_features)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.rf.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.rf.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha

    def compute_bundle_weight(self, all_sigma_feature, rel_density_grid_feature, app_mask, ray_valid, dists):
        n_samples = ray_valid.shape[1]
        bundle_sigma = all_sigma_feature[..., None, None].repeat(1, 1, self.bundle_size, self.bundle_size)
        # ic(1, bundle_sigma[app_mask], bundle_sigma[app_mask].shape, rel_density_grid_feature.shape)

        # bundle_sigma[app_mask] = F.softplus(bundle_sigma[app_mask] + rel_density_grid_feature, beta=10)
        # bundle_sigma[app_mask] = F.softplus(bundle_sigma[app_mask], beta=10)
        # bundle_sigma[app_mask] = F.relu(bundle_sigma[app_mask] + rel_density_grid_feature)

        bundle_sigma[app_mask] += rel_density_grid_feature
        bundle_sigma[ray_valid] = self.feature2density(bundle_sigma[ray_valid])

        # ic(2, bundle_sigma[app_mask], rel_density_grid_feature)
        # ic(bundle_weight[app_mask][:10], rel_density_grid[:10], rel_density, extra)

        # reshape to align with how raw2alpha expects
        bundle_sigma = bundle_sigma.permute(0, 2, 3, 1).reshape(-1, n_samples)
        bundle_dists = dists.reshape(-1, 1, n_samples).repeat(1, self.bundle_size**2, 1).reshape(-1, n_samples)
        _, bundle_weight, _ = raw2alpha(bundle_sigma, bundle_dists * self.distance_scale)
        # reshape back to original
        bundle_weight = bundle_weight.reshape(-1, self.bundle_size, self.bundle_size, n_samples).permute(0, 3, 1, 2)
        return bundle_weight

    def forward(self, rays_chunk, focal, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        # rays_chunk: (N, 6)

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, focal, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, focal, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        # xyz_sampled_shape: (N, N_samples, 3+1)
        # z_vals.shape: (N, N_samples)
        # ray_valid.shape: (N, N_samples)
        # ic(z_vals, z_vals/focal, z_vals.shape, xyz_sampled_shape)
        xyz_sampled_shape = xyz_sampled[:, :, :3].shape

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled_shape)
        rays_up = rays_chunk[:, 6:9]
        rays_o = rays_chunk[:, :3]
        rays_up = rays_up.view(-1, 1, 3).expand(xyz_sampled_shape)
        n_samples = xyz_sampled_shape[1]
        
        if self.alphaMask is not None and False:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid
            all_alphas = self.alphaMask.sample_alpha(xyz_sampled).detach().cpu().reshape(xyz_sampled_shape[0], xyz_sampled.shape[1])


        # sigma.shape: (N, N_samples)
        sigma = torch.zeros(xyz_sampled_shape[:-1], device=xyz_sampled.device)
        world_normal = torch.zeros(xyz_sampled_shape, device=xyz_sampled.device)
        all_sigma_feature = torch.zeros(xyz_sampled_shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled_shape[:2], self.bundle_size, self.bundle_size, 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.rf.normalize_coord(xyz_sampled)
            sigma_feature, normal_feature = self.rf.compute_density_norm(xyz_sampled[ray_valid], self.feature2density)

            validsigma = self.feature2density(sigma_feature)
            all_sigma_feature[ray_valid] = sigma_feature
            sigma[ray_valid] = validsigma
            world_normal[ray_valid] = normal_feature

        # xyz_c = xyz_sampled.detach().cpu()
        # fig = px.scatter_3d(x=xyz_c[:64, :, 0].flatten(), y=xyz_c[:64, :, 1].flatten(), z=xyz_c[:64, :, 2].flatten(), color=sigma.detach().cpu()[:64].flatten())
        # fig.show()
        # assert(False)


        # weight: [N_rays, N_samples]
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        # app stands for appearance
        app_mask = weight > self.rayMarch_weight_thres
        rgb_app_mask = app_mask[:, :, None, None].repeat(1, 1, self.bundle_size, self.bundle_size)
        bundle_size_w = z_vals / focal * (self.bundle_size-1)
        v_normal_all = torch.zeros((*xyz_sampled_shape[:2], 3), device=xyz_sampled.device)
        bundle_weight = weight[..., None, None].repeat(1, 1, self.bundle_size, self.bundle_size)

        d_world_normal_map = torch.sum(weight[..., None] * world_normal, 1)
        d_refdirs = viewdirs - 2 * (viewdirs * world_normal).sum(-1, keepdim=True) * world_normal

        if app_mask.any():
            app_features = self.rf.compute_appfeature(xyz_sampled[app_mask])
            if self.has_grid:
                # bundle_size_w: (N, 1)
                valid_rgbs, rel_density_grid_feature, roughness, normal = self.renderModule(
                    xyz_sampled[app_mask], viewdirs[app_mask], app_features, bundle_size_w[app_mask][:, None], rays_up[app_mask], refdirs=d_refdirs[app_mask])
                v_normal_all[app_mask] = normal / torch.norm(normal, dim=-1, keepdim=True)

                bundle_weight = self.compute_bundle_weight(all_sigma_feature, rel_density_grid_feature, app_mask, ray_valid, dists)
            else:
                valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features, refdirs=d_refdirs[app_mask])
            rgb[rgb_app_mask] = valid_rgbs.reshape(-1, 3)

        acc_map = torch.sum(bundle_weight, 1)

        # calculate depth
        bundle_z_vals = z_vals[..., None, None].repeat(1, 1, self.bundle_size, self.bundle_size)
        bundle_rays_chunk = rays_chunk[:, None, None, :].repeat(1, self.bundle_size, self.bundle_size, 1)
        depth_map = torch.sum(bundle_weight * bundle_z_vals, 1)
        # (N, bundle_size, bundle_size)
        depth_map = depth_map + (1. - acc_map) * bundle_rays_chunk[..., -1]

        # view dependent normal map
        # N, 3, 3
        row_basis = torch.stack([
            -torch.cross(viewdirs[:, 0], rays_up[:, 0]),
            viewdirs[:, 0],
            rays_up[:, 0],
        ], dim=1)
        # ic(row_basis[0])
        d_normal_map = torch.matmul(row_basis, d_world_normal_map.unsqueeze(-1)).squeeze(-1)
        # d_normal_map = rays_up[:, 0] / rays_up[:, 0].norm(dim=-1, keepdim=True)

        inds = bundle_weight[:, :, self.bundle_size//2, self.bundle_size//2].max(dim=1).indices
        xyz = xyz_sampled[range(xyz_sampled_shape[0]), inds]#.cpu().numpy()
        # ref_mask = reflectivity > 0.1
        if self.bundle_size == 3 and app_mask.any():
            # v_normal_map = self.compute_normal(depth_map, focal)
            v_normal_map = torch.sum(weight[..., None] * v_normal_all, 1)

            l = 0
            l_normal_map = l*v_normal_map + (1-l)*d_normal_map
            # v_normal_map = v_normal_map / v_normal_map.norm(dim=-1, keepdim=True)

            # project normal map from camera space to world space
            # world_normal.shape: (N, 3)
            # v_world_normal_map = v_normal_map[..., 0:1] * z_basis + v_normal_map[..., 1:2] * rays_up[:, 0] + v_normal_map[..., 2:3] * -viewdirs[:, 0]
            # v_world_normal_map = v_normal_map[..., 1:2] * rays_up[:, 0]# + v_normal_map[..., 2:3] * -viewdirs[:, 0]
            v_world_normal_map = torch.matmul(row_basis.permute(0, 2, 1), v_normal_map.unsqueeze(-1)).squeeze(-1)
            l_world_normal_map = torch.matmul(row_basis.permute(0, 2, 1), l_normal_map.unsqueeze(-1)).squeeze(-1)

            normal_sim = (v_normal_map * d_normal_map.detach()).sum(dim=1)
            normal_sim = normal_sim.clamp(max=self.max_normal_similarity).mean()
            normal_sim = 0

            # right_vec = (z_basis).cpu()
            # up_vec = (rays_up[:, 0]).cpu()
            # front_vec = (-viewdirs[:, 0]).cpu()
            # v_world_normal = world_normal.cpu()
            # v_normal_map = normal_map.cpu()

            # g1 = go.Scatter3d(x=right_vec[:, 0], y=right_vec[:, 1], z=right_vec[:, 2], marker=dict(color='red'), mode='markers')
            # g3 = go.Scatter3d(x=front_vec[:, 0], y=front_vec[:, 1], z=front_vec[:, 2], marker=dict(color='blue'), mode='markers')
            # g2 = go.Scatter3d(x=up_vec[:, 0], y=up_vec[:, 1], z=up_vec[:, 2], marker=dict(color='green'), mode='markers')
            # g4 = go.Scatter3d(x=v_world_normal[:, 0], y=v_world_normal[:, 1], z=v_world_normal[:, 2], marker=dict(color='purple'), mode='markers')
            # g5 = go.Scatter3d(x=v_normal_map[:, 0], y=v_normal_map[:, 1], z=v_normal_map[:, 2], marker=dict(color='orange'), mode='markers')

            # fig = go.Figure(data=[g1, g2, g3, g4, g5])
            # fig.show()
            # assert(False)

            # reflected vector: r = d - 2(d*n)n
            # d = (0, 0, 1) relative to the normal
            if self.enable_reflections:
                refdirs = viewdirs - 2 * (viewdirs * l_world_normal_map[:, None]).sum(-1, keepdim=True) * l_world_normal_map[:, None]

                ref_valid_rgbs = self.reflectionModule(xyz_sampled[app_mask], refdirs[app_mask], app_features, bundle_size_w=bundle_size_w[app_mask][:, None], ray_up=rays_up[app_mask], roughness=roughness)
                rgb[rgb_app_mask] = ref_valid_rgbs.reshape(-1, 3)
        else:
            v_world_normal_map = d_world_normal_map
            v_normal_map = d_normal_map
            l_normal_map = d_normal_map
            normal_sim = 0

        rgb_map = torch.sum(bundle_weight[..., None] * rgb, 1)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        # return rgb_map, depth_map, d_world_normal_map, acc_map, xyz, normal_sim
        return rgb_map, depth_map, l_normal_map, acc_map, xyz, normal_sim

