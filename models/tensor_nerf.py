import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from icecream import ic

from . import render_modules
import plotly.express as px
import plotly.graph_objects as go
import random
import hydra

from .tensoRF import TensorCP, TensorVM, TensorVMSplit
from .multi_level_rf import MultiLevelRF
from torch.autograd import grad
import matplotlib.pyplot as plt

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([
        torch.ones(alpha.shape[0], 1, device=alpha.device),
        1. - alpha + 1e-10
    ], dim=-1), dim=-1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


class AlphaGridMask(torch.nn.Module):
    def __init__(self, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.register_buffer('aabb', aabb)

        aabbSize = self.aabb[1] - self.aabb[0]
        invgrid_size = 1.0/aabbSize * 2
        alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        grid_size = torch.LongTensor(
            [alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]])
        self.register_buffer('grid_size', grid_size)
        self.register_buffer('invgrid_size', invgrid_size)
        self.register_buffer('alpha_volume', alpha_volume)

    def sample_alpha(self, xyz_sampled, contract_space=False):
        xyz_sampled = self.normalize_coord(xyz_sampled, contract_space)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled[..., :3].view(
            1, -1, 1, 1, 3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled, contract_space):
        coords = (xyz_sampled[..., :3]-self.aabb[0]) * self.invgrid_size - 1
        size = xyz_sampled[..., 3:4]
        normed = torch.cat((coords, size), dim=-1)
        if contract_space:
            dist = torch.linalg.norm(normed[..., :3], dim=-1, keepdim=True, ord=torch.inf) + 1e-8
            direction = normed[..., :3] / dist
            contracted = torch.where(dist > 1, (2-1/dist), dist)/2 * direction
            return torch.cat([ contracted, xyz_sampled[..., 3:] ], dim=-1)
        else:
            return normed


class TensorNeRF(torch.nn.Module):
    def __init__(self, rf, grid_size, aabb, diffuse_module, normal_module=None, ref_module=None, bg_module=None,
                 alphaMask=None, near_far=[2.0, 6.0], nEnvSamples=100, specularity_threshold=0.1, max_recurs=0,
                 max_normal_similarity=1, infinity_border=False,
                 density_shift=-10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                 fea2denseAct='softplus'):
        super(TensorNeRF, self).__init__()
        self.rf = rf(aabb=aabb, grid_size=grid_size)
        self.ref_module = ref_module(in_channels=self.rf.app_dim) if ref_module is not None else None
        self.normal_module = normal_module(in_channels=self.rf.app_dim) if normal_module is not None else None
        self.diffuse_module = diffuse_module(in_channels=self.rf.app_dim)
        self.bg_module = bg_module

        self.alphaMask = alphaMask
        self.infinity_border = infinity_border

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far

        self.max_recurs = max_recurs
        self.specularity_threshold = specularity_threshold
        self.max_bounce_rays = 100
        self.nEnvSamples = nEnvSamples

        f_blur = torch.tensor([1, 2, 1]) / 4
        f_edge = torch.tensor([-1, 0, 1]) / 2
        self.register_buffer('f_blur', f_blur)
        self.register_buffer('f_edge', f_edge)

        self.max_normal_similarity = max_normal_similarity
        self.l = 0
        
    @property
    def device(self):
        return self.rf.units.device

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        grad_vars = []
        grad_vars += self.rf.get_optparam_groups(
            lr_init_spatial, lr_init_network)
        if isinstance(self.ref_module, torch.nn.Module):
            grad_vars += [{'params': self.ref_module.parameters(),
                           'lr': lr_init_network}]
        if hasattr(self, 'normal_module') and isinstance(self.normal_module, torch.nn.Module):
            grad_vars += [{'params': self.normal_module.parameters(),
                           'lr': lr_init_network*2}]
        if hasattr(self, 'diffuse_module') and isinstance(self.diffuse_module, torch.nn.Module):
            grad_vars += [{'params': self.diffuse_module.parameters(),
                           'lr': lr_init_network*2}]
        if hasattr(self, 'bg_module') and isinstance(self.bg_module, torch.nn.Module):
            grad_vars += [{'params': self.bg_module.parameters(),
                           'lr': lr_init_network*2}]
        return grad_vars

    def save(self, path, config):
        ckpt = {'config': config, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update(
                {'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    @staticmethod
    def load(ckpt):
        config = ckpt['config']
        aabb = ckpt['state_dict']['rf.aabb']
        # ic(ckpt['state_dict'].keys())
        grid_size = ckpt['state_dict']['rf.grid_size'].cpu()
        rf = hydra.utils.instantiate(config)(aabb=aabb, grid_size=grid_size)
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[
                                            :length].reshape(ckpt['alphaMask.shape']))
            rf.alphaMask = AlphaGridMask(
                ckpt['alphaMask.aabb'], alpha_volume.float())
        rf.load_state_dict(ckpt['state_dict'], strict=False)
        return rf

    def sample_ray_ndc(self, rays_o, rays_d, focal, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.rf.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            l = torch.rand_like(interpx)
            interpx += l.to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.rf.aabb[0] > rays_pts) | (
            rays_pts > self.rf.aabb[1])).any(dim=-1)

        # add size
        rays_pts = torch.cat([rays_pts, interpx.unsqueeze(-1)/focal], dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, focal, is_train=True, N_samples=-1, N_env_samples=-1):
        # focal: ratio of meters to pixels at a distance of 1 meter
        N_samples = N_samples if N_samples > 0 else self.rf.nSamples
        N_env_samples = N_env_samples if N_env_samples > 0 else self.nEnvSamples
        stepsize = self.rf.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.rf.aabb[1].to(rays_o) - rays_o) / vec
        rate_b = (self.rf.aabb[0].to(rays_o) - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples, device=rays_o.device)[None].float()
        # extend rng to sample towards infinity
        if N_env_samples > 0:
            ext_rng = N_samples + N_env_samples / \
                torch.linspace(1, 1/N_env_samples, N_env_samples,
                               device=rays_o.device)[None].float()
            rng = torch.cat([rng, ext_rng], dim=1)

        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            # N, N_samples
            # add noise along each ray
            brng = rng.reshape(-1, N_samples+N_env_samples)
            # brng = brng + torch.rand_like(brng[:, [0], [0]])
            # r = torch.rand_like(brng[:, 0:1, 0:1])
            r = torch.rand_like(brng[:, 0:1])
            brng = brng + r
            rng = brng.reshape(-1, N_samples+N_env_samples)
        step = stepsize * rng
        interpx = (t_min[..., None] + step)

        rays_pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.rf.aabb[0] > rays_pts) | (
            rays_pts > self.rf.aabb[1])).any(dim=-1)

        # add size
        rays_pts = torch.cat([rays_pts, interpx.unsqueeze(-1)/focal], dim=-1)
        env_mask = torch.zeros_like(mask_outbbox)
        env_mask[:, N_samples:] = 1

        if self.rf.contract_space:
            mask_outbbox = torch.zeros_like(mask_outbbox)

        return rays_pts, interpx, ~mask_outbbox, env_mask

    @torch.no_grad()
    def getDenseAlpha(self, grid_size=None):
        grid_size = self.rf.grid_size if grid_size is None else grid_size

        dense_xyz = torch.stack([*torch.meshgrid(
            torch.linspace(-1, 1, grid_size[0]),
            torch.linspace(-1, 1, grid_size[1]),
            torch.linspace(-1, 1, grid_size[2])),
            torch.ones((grid_size[0], grid_size[1],
                       grid_size[2]))*self.rf.units.min().cpu()*0.5
        ], -1).to(self.device)

        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(grid_size[0]):
            xyz_norm = dense_xyz[i].view(-1, 4)
            sigma_feature = self.rf.compute_densityfeature(xyz_norm)
            sigma = self.feature2density(sigma_feature)
            alpha[i] = 1 - torch.exp(-sigma*self.rf.stepSize).reshape(*alpha[i].shape)

        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, grid_size=(200, 200, 200)):

        grid_size = [int(self.rf.density_res_multi*g) for g in grid_size]
        alpha, dense_xyz = self.getDenseAlpha(grid_size)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = grid_size[0] * grid_size[1] * grid_size[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks,
                             padding=ks // 2, stride=1).view(grid_size[::-1])
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.rf.aabb, alpha).to(self.device)

        valid_xyz = dense_xyz[alpha > 0.0]
        if valid_xyz.shape[0] < 1:
            print("No volume")
            return self.rf.aabb

        xyz_min = valid_xyz.amin(0)[:3]
        xyz_max = valid_xyz.amax(0)[:3]

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" %
              (total/total_voxels*100))
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
                vec = torch.where(
                    rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.rf.aabb[1].to(rays_o) - rays_o) / vec
                rate_b = (self.rf.aabb[0].to(rays_o) - rays_o) / vec
                # .clamp(min=near, max=far)
                t_min = torch.minimum(rate_a, rate_b).amax(-1)
                # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _, _ = self.sample_ray(
                    rays_o, rays_d, focal, N_samples=N_samples, is_train=False)
                # Issue: calculate size
                mask_inbbox = (self.alphaMask.sample_alpha(
                    xyz_sampled).view(xyz_sampled.shape[:-1]) > 0, self.rf.contract_space).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(
            f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered], mask_filtered

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus_shift":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "softplus":
            return F.softplus(density_features)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        elif self.fea2denseAct == "relu_shift":
            return F.relu(density_features+self.density_shift)

    def sample_occupied(self):  # , rays_chunk, ndc_ray=False, N_samples=-1):
        # viewdirs = rays_chunk[:, 3:6]
        # if ndc_ray:
        #     xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, 1, is_train=True,N_samples=N_samples)
        #     dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        #     rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        #     dists = dists * rays_norm
        #     viewdirs = viewdirs / rays_norm
        # else:
        #     xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, 1, is_train=True,N_samples=N_samples)
        #     dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        samps = torch.rand((10000, 4), device=self.device)*2 - 1
        sigma_feature = self.rf.compute_densityfeature(samps)
        validsigma = self.feature2density(sigma_feature).squeeze()
        mask = validsigma > validsigma.mean()
        inds, = torch.where(mask)
        ind = random.randint(0, len(inds))
        xyz = samps[inds[ind]]
        return xyz

    def shrink(self, new_aabb, voxel_size):
        return
        if self.rf.contract_space:
            return
        else:
            self.rf.shrink(new_aabb, voxel_size)

    def recover_envmap(self, res, xyz=None):
        if xyz is None:
            xyz = self.sample_occupied()

        app_feature = self.rf.compute_appfeature(xyz.reshape(1, -1))
        B = 2*res*res
        staticdir = torch.zeros((B, 3), device=self.device)
        staticdir[:, 0] = 1
        ele_grid, azi_grid = torch.meshgrid(
            torch.linspace(-np.pi/2, np.pi/2, res, dtype=torch.float32),
            torch.linspace(-np.pi, np.pi, 2*res, dtype=torch.float32), indexing='ij')
        # each col of x ranges from -pi/2 to pi/2
        # each row of y ranges from -pi to pi
        app_features = app_feature.reshape(
            1, -1).expand(B, app_feature.shape[-1])
        xyz_samp = xyz.reshape(1, -1).expand(B, xyz.shape[-1])
        ang_vecs = torch.stack([
            torch.cos(ele_grid) * torch.cos(azi_grid),
            torch.cos(ele_grid) * torch.sin(azi_grid),
            -torch.sin(ele_grid),
        ], dim=-1).to(self.device)
        # roughness = 1/np.pi*torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
        roughness = 20 * \
            torch.ones((app_features.shape[0], 1),
                       dtype=xyz.dtype, device=xyz.device)
        envmap = self.ref_module(xyz_samp, staticdir, app_features, refdirs=ang_vecs.reshape(
            -1, 3), roughness=roughness).reshape(res, 2*res, 3)
        color = self.ref_module(xyz_samp, ang_vecs.reshape(-1, 3), app_features,
                                refdirs=staticdir, roughness=roughness).reshape(res, 2*res, 3)
        return envmap, color

    def at_infinity(self, xyz_sampled, max_dist=10):
        margin = 1 - 1/max_dist/2
        at_infinity = torch.linalg.norm(
            xyz_sampled, dim=-1, ord=torch.inf).abs() >= margin
        return at_infinity

    def forward(self, rays_chunk, focal, recur=0, output_alpha=None, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        # rays_chunk: (N, 6)

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3], viewdirs, focal, is_train=is_train, N_samples=N_samples)
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid, env_mask = self.sample_ray(
                rays_chunk[:, :3], viewdirs, focal, is_train=is_train, N_samples=N_samples)
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        # xyz_sampled_shape: (N, N_samples, 3+1)
        # z_vals.shape: (N, N_samples)
        # ray_valid.shape: (N, N_samples)
        # ic(z_vals, z_vals/focal, z_vals.shape, xyz_sampled_shape)
        xyz_sampled_shape = xyz_sampled[:, :, :3].shape

        xyz_normed = self.rf.normalize_coord(xyz_sampled)

        device = xyz_sampled.device

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled_shape)
        rays_o = rays_chunk[:, :3]
        rays_up = rays_chunk[:, 6:9]
        rays_up = rays_up.view(-1, 1, 3).expand(xyz_sampled_shape)
        B = xyz_sampled.shape[0]
        n_samples = xyz_sampled_shape[1]

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(
                xyz_sampled[ray_valid], contract_space=self.rf.contract_space)
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        # sigma.shape: (N, N_samples)
        sigma = torch.zeros(xyz_sampled_shape[:-1], device=device)
        world_normal = torch.zeros(
            xyz_sampled_shape, device=device)
        rgb = torch.zeros(
            (*xyz_sampled_shape[:2], 3), device=device)
        p_world_normal = torch.zeros(
            xyz_sampled_shape, device=device)
        # p_world_normal = world_normal.clone()

        if ray_valid.any():

            sigma_feature = self.rf.compute_densityfeature(xyz_normed[ray_valid])
            #  sigma_feature, world_normal[ray_valid] = self.rf.compute_density_norm(xyz_normed[ray_valid], self.feature2density)
            #  _, world_normal[ray_valid] = self.rf.compute_density_norm(xyz_normed[ray_valid], self.feature2density)
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

            with torch.enable_grad():
                xyz_g = xyz_sampled[ray_valid].clone()
                xyz_g.requires_grad = True

                # compute sigma
                xyz_g_normed = self.rf.normalize_coord(xyz_g)
                sigma_feature = self.rf.compute_densityfeature(xyz_g_normed)
                validsigma = self.feature2density(sigma_feature)
                sigma[ray_valid] = validsigma

                # compute normal
                grad_outputs = torch.ones_like(validsigma)
                surf_grad = grad(validsigma, xyz_g, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0][:, :3]
                surf_grad = surf_grad / (torch.norm(surf_grad, dim=1, keepdim=True)+1e-8)

                world_normal[ray_valid] = surf_grad


        if self.rf.contract_space and self.infinity_border:
            at_infinity = self.at_infinity(xyz_normed)
            sigma[at_infinity] = 100

        # weight: [N_rays, N_samples]
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)
        # floater_loss = torch.matmul(weight.reshape(
        #     B, -1, 1), weight.reshape(B, 1, -1)).sum(dim=-1).sum(dim=-1).mean()
        floater_loss = 0

        # app stands for appearance
        app_mask = (weight > self.rayMarch_weight_thres)

        if app_mask.any():
            #  Compute normals for app mask
            #  _, normal_feature = self.rf.compute_density_norm(xyz_normed[app_mask], self.feature2density)
            #  _, normal_feature = self.rf.compute_density_norm(xyz_normed[app_mask], self.feature2density)
            #  world_normal[app_mask] = normal_feature
            #  a = viewdirs[app_mask][..., :3]
            #  world_normal[app_mask] = a / torch.linalg.norm(a, dim=-1, keepdim=True)

            #  with torch.enable_grad():
            #      xyz_g = xyz_sampled[app_mask].clone()
            #      xyz_g.requires_grad = True
            #
            #      # compute sigma
            #      xyz_g_normed = self.rf.normalize_coord(xyz_g)
            #      sigma_feature = self.rf.compute_densityfeature(xyz_g_normed)
            #      validsigma = self.feature2density(sigma_feature)
            #
            #      # compute normal
            #      grad_outputs = torch.ones_like(validsigma)
            #      g = grad(validsigma, xyz_g, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)
            #      world_normal[app_mask] = g[0][:, :3]

            # get base color of the point
            app_features = self.rf.compute_appfeature(xyz_normed[app_mask])
            diffuse, tint, roughness = self.diffuse_module(
                xyz_normed[app_mask], viewdirs[app_mask], app_features)
            
            # handle reflections
            if self.ref_module is not None:
                p_world_normal[app_mask] = self.normal_module(xyz_normed[app_mask], app_features)

                l = self.l if is_train else 0
                v_world_normal = (1-l)*p_world_normal + l*world_normal
                v_world_normal = v_world_normal / \
                    (v_world_normal.norm(dim=-1, keepdim=True) + 1e-8)

                d_refdirs = viewdirs - 2 * \
                    (viewdirs * v_world_normal).sum(-1, keepdim=True) * v_world_normal

                # roughness = 1/np.pi*torch.ones((app_features.shape[0], 1), dtype=xyz_normed.dtype, device=xyz_normed.device)

                # valid_rgbs = valid_rgbs.reshape(-1, 3)
                M = diffuse.shape[0]
                ref_rgbs = torch.zeros_like(diffuse)
                bounce_mask = torch.zeros((M), dtype=bool, device=device)
                if recur < self.max_recurs:
                    # decide how many bounces to calculate
                    prob = torch.linalg.norm(tint, dim=-1)
                    n_bounces = min(min(self.max_bounce_rays, M),
                                    (prob > self.specularity_threshold).sum())
                    inds = torch.argsort(-prob)[:n_bounces]
                    bounce_mask[inds] = 1
                    # d_refdirs must be detached otherwise it causes an error. TODO
                    bounce_rays = torch.cat([
                        xyz_sampled[app_mask][bounce_mask][..., :3],
                        d_refdirs[app_mask][bounce_mask].detach(),
                        rays_up[app_mask][bounce_mask]
                    ], dim=-1)
                    rec_rgbs, rec_depth_map, _, rec_acc_map, rec_termination_xyz, _, _ = self(
                        bounce_rays, focal, recur=recur+1, white_bg=white_bg, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples)
                    ref_rgbs[bounce_mask] = rec_rgbs[:, 0, 0, :]
                if (~bounce_mask).any():
                    ref_rgbs[~bounce_mask] = self.ref_module(
                        xyz_normed[app_mask][~bounce_mask], viewdirs[app_mask][~bounce_mask], app_features[~bounce_mask],
                        refdirs=d_refdirs[app_mask][~bounce_mask], roughness=roughness[~bounce_mask])
                rgb[app_mask] = (tint * ref_rgbs + diffuse).type(rgb.dtype)
                # rgb[app_mask] = torch.linalg.norm(diffuse, dim=-1, keepdim=True).type(rgb.dtype)
            else:
                rgb[app_mask] = diffuse
            # rgb[rgb_app_mask] = valid_rgbs
        else:
            v_world_normal = world_normal
            roughness = torch.tensor(0.0)

        acc_map = torch.sum(weight, 1)

        normal_sim = -(p_world_normal * world_normal).sum(dim=-1).clamp(max=self.max_normal_similarity)
        normal_sim = (weight * normal_sim).mean()

        # calculate depth

        with torch.no_grad():
            # shadow_map = torch.sum(weight * shadows, 1)
            depth_map = torch.sum(weight * z_vals, 1)
            # (N, bundle_size, bundle_size)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

            # view dependent normal map
            # N, 3, 3
            row_basis = torch.stack([
                -torch.cross(viewdirs[:, 0], rays_up[:, 0]),
                viewdirs[:, 0],
                rays_up[:, 0],
            ], dim=1)
            p_world_normal_map = torch.sum(
                weight[..., None] * p_world_normal, 1)
            p_world_normal_map = p_world_normal_map / \
                (torch.norm(p_world_normal_map, dim=-1, keepdim=True)+1e-8)
            d_world_normal_map = torch.sum(weight[..., None] * world_normal, 1)
            v_world_normal_map = torch.sum(weight[..., None] * v_world_normal, 1)
            d_normal_map = torch.matmul(row_basis, d_world_normal_map.unsqueeze(-1)).squeeze(-1)
            # p_normal_map = torch.matmul(
            #     row_basis, p_world_normal_map.unsqueeze(-1)).squeeze(-1)
            v_normal_map = torch.matmul(row_basis, v_world_normal_map.unsqueeze(-1)).squeeze(-1)

            # # extract
            # inds = weight.max(dim=1).indices
            # termination_xyz = xyz_sampled[range(xyz_sampled_shape[0]), inds]

        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if self.bg_module is not None:
            bg = self.bg_module(viewdirs[:, 0, :])
            # rgb_map = acc_map[..., None] * rgb_map + (1. - acc_map[..., None]) * bg.reshape(-1, 1, 1, 3)
            rgb_map = rgb_map + \
                (1. - acc_map[..., None]) * bg.reshape(-1, 3)
        else:
            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1. - acc_map[..., None])
        # rgb_map = bg.reshape(-1, 1, 1, 3)

        rgb_map = rgb_map.clamp(0, 1)

        # return rgb_map, depth_map, p_world_normal_map, acc_map, termination_xyz, normal_sim
        # return rgb_map, depth_map, p_normal_map, acc_map, termination_xyz, normal_sim+floater_loss, roughness.mean().cpu()
        # return rgb_map, depth_map, None, acc_map, None, normal_sim+floater_loss, roughness.mean().cpu()
        return rgb_map, depth_map, v_normal_map, acc_map, None, normal_sim+floater_loss, roughness.mean().cpu()
