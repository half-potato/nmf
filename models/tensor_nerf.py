import torch
import torch.nn.functional as F
import numpy as np
import time
from icecream import ic

from . import render_modules
from .tonemap import SRGBTonemap
import plotly.express as px
import plotly.graph_objects as go
import random
import hydra

from torch.autograd import grad
import matplotlib.pyplot as plt
import math
from .logger import Logger
import utils

LOGGER = Logger(enable=False)


def raw2alpha(sigma, flip, dist):
    # sigma, dist  [N_rays, N_samples]
    v = torch.exp(-sigma*dist)
    alpha = torch.where(flip, v, 1-v)
    # alpha = 1. - torch.exp(-sigma*dist)

    # T is the term that integrates the alpha backwards to prevent occluded objects from influencing things
    # multiply in exponential space to take exponential of integral
    T = torch.cumprod(torch.cat([
        torch.ones(alpha.shape[0], 1, device=alpha.device),
        1. - alpha + 1e-10
    ], dim=-1), dim=-1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


class TensorNeRF(torch.nn.Module):
    def __init__(self, rf, grid_size, aabb, diffuse_module, sampler=None, brdf=None, tonemap=None, normal_module=None, ref_module=None, bg_module=None,
                 alphaMask=None, near_far=[2.0, 6.0], nEnvSamples=100, specularity_threshold=0.005, max_recurs=0,
                 max_normal_similarity=1, infinity_border=False, min_refraction=1.1, enable_refraction=True,
                 density_shift=-10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                 max_bounce_rays=4000, roughness_rays=3, bounce_min_weight=0.001, appdim_noise_std=0.0,
                 world_bounces=0, fea2denseAct='softplus', enable_alpha_mask=True, selector=None,
                 max_floater_loss=6, **kwargs):
        super(TensorNeRF, self).__init__()
        self.rf = rf(aabb=aabb, grid_size=grid_size)
        self.ref_module = ref_module(in_channels=self.rf.app_dim) if ref_module is not None else None
        self.normal_module = normal_module(in_channels=self.rf.app_dim) if normal_module is not None else None
        self.diffuse_module = diffuse_module(in_channels=self.rf.app_dim)
        self.brdf = brdf(in_channels=self.rf.app_dim) if brdf is not None else None
        self.sampler = sampler if sampler is None else sampler(num_samples=roughness_rays)
        self.selector = selector
        self.bg_module = bg_module
        if tonemap is None:
            self.tonemap = SRGBTonemap()
        else:
            self.tonemap = tonemap

        self.world_bounces = world_bounces
        self.alphaMask = alphaMask
        self.infinity_border = infinity_border
        self.enable_alpha_mask = enable_alpha_mask
        self.max_floater_loss = max_floater_loss
        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct
        self.appdim_noise_std = appdim_noise_std

        self.near_far = near_far
        self.nEnvSamples = nEnvSamples
        self.bounce_min_weight = bounce_min_weight
        self.min_refraction = min_refraction
        self.enable_refraction = enable_refraction
        self.roughness_rays = roughness_rays
        self.max_recurs = max_recurs
        self.specularity_threshold = specularity_threshold
        self.max_bounce_rays = max_bounce_rays


        f_blur = torch.tensor([1, 2, 1]) / 4
        f_edge = torch.tensor([-1, 0, 1]) / 2
        self.register_buffer('f_blur', f_blur)
        self.register_buffer('f_edge', f_edge)

        self.max_normal_similarity = max_normal_similarity
        self.l = 0
        
    @property
    def device(self):
        return self.rf.units.device

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001, lr_bg=0.025, lr_scale=1):
        grad_vars = []
        # TODO REMOVE
        grad_vars += self.rf.get_optparam_groups(lr_init_spatial, lr_init_network)
        if isinstance(self.normal_module, torch.nn.Module):
            grad_vars += [{'params': self.normal_module.parameters(),
                           'lr': self.normal_module.lr*lr_scale}]
        if self.ref_module is not None:
            grad_vars += [{'params': list(self.ref_module.parameters()), 'lr': lr_scale*self.ref_module.lr}]
        if isinstance(self.diffuse_module, torch.nn.Module):
            grad_vars += [{'params': self.diffuse_module.parameters(),
                           'lr': lr_init_network}]
        if isinstance(self.brdf, torch.nn.Module):
            grad_vars += [{'params': self.brdf.parameters(),
                           'lr': self.brdf.lr}]
        # TODO REMOVE
        if isinstance(self.bg_module, torch.nn.Module):
            grad_vars += [{'params': self.bg_module.parameters(),
                'lr': self.bg_module.lr, 'name': 'bg'}]
        return grad_vars

    def save(self, path, config):
        print(f"Saving nerf to {path}")
        if self.bg_module is not None:
            config['bg_module']['bg_resolution'] = self.bg_module.bg_resolution
        ckpt = {'config': config, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.cpu()
            ckpt.update({'alphaMask': alpha_volume})
            #  alpha_volume = self.alphaMask.alpha_volume.cpu().numpy()
            #  ckpt.update({'alphaMask.shape': alpha_volume.shape})
            #  ckpt.update(
            #      {'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    @staticmethod
    def load(ckpt, **kwargs):
        config = ckpt['config']
        aabb = ckpt['state_dict']['rf.aabb']
        # ic(ckpt['state_dict'].keys())
        grid_size = ckpt['state_dict']['rf.grid_size'].cpu()
        rf = hydra.utils.instantiate(config)(aabb=aabb, grid_size=grid_size)
        if 'alphaMask.aabb' in ckpt.keys():
            #  length = np.prod(ckpt['alphaMask.shape'])
            #  alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[
            #                                  :length].reshape(ckpt['alphaMask.shape'])).float()
            alpha_volume = ckpt['alphaMask']
            rf.alphaMask = utils.AlphaGridMask(
                ckpt['alphaMask.aabb'], alpha_volume)
        rf.load_state_dict(ckpt['state_dict'], **kwargs)
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

    def sample_ray(self, rays_o, rays_d, focal, is_train=True, override_near=None, N_samples=-1, N_env_samples=-1):
        # focal: ratio of meters to pixels at a distance of 1 meter
        N_samples = N_samples if N_samples > 0 else self.rf.nSamples
        N_env_samples = N_env_samples if N_env_samples > 0 else self.nEnvSamples
        stepsize = self.rf.stepSize
        near, far = self.near_far
        if override_near is not None:
            near = override_near
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

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.rf.aabb[0] > rays_pts) | (rays_pts > self.rf.aabb[1])).any(dim=-1)

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
        # alpha[alpha >= self.alphaMask_thres] = 1
        # alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = utils.AlphaGridMask(self.rf.aabb, alpha > self.alphaMask_thres).to(self.device)

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
                mask_inbbox = self.alphaMask.sample_alpha(
                        xyz_sampled).reshape(xyz_sampled.shape[:-1]).any(-1)

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
        elif self.fea2denseAct == "relu_shift":
            return F.relu(density_features+self.density_shift)
        elif self.fea2denseAct == "identity":
            return density_features

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
        if self.rf.contract_space:
            return
        else:
            self.rf.shrink(new_aabb, voxel_size)

    def render_env_sparse(self, ray_origins, env_dirs, roughness: float):
        B, M = env_dirs.shape[:2]
        ray_origins = torch.cat([ray_origins, roughness*torch.ones((B, 1), device=self.device)], dim=-1)
        norm_ray_origins = self.rf.normalize_coord(ray_origins)
        app_features = self.rf.compute_appfeature(norm_ray_origins)
        app_features = app_features.reshape(B, 1, -1).expand(B, M, -1)
        norm_ray_origins = norm_ray_origins.reshape(B, 1, -1).expand(B, M, -1)
        roughness = torch.tensor(roughness, device=ray_origins.device)
        staticdir = torch.zeros((B*M, 3), device=self.device)
        staticdir[:, 0] = 1
        color = self.ref_module(
                pts=norm_ray_origins.reshape(B*M, -1),
                features=app_features.reshape(B*M, -1),
                refdirs=env_dirs.reshape(-1, 3),
                viewdirs=staticdir,
                roughness=roughness)
        return color

    def recover_envmap(self, res, xyz=None, roughness=None):
        if xyz is None:
            xyz = self.sample_occupied()

        app_feature = self.rf.compute_appfeature(xyz.reshape(1, -1))
        B = 2*res*res
        staticdir = torch.zeros((B, 3), device=self.device)
        staticdir[:, 0] = 1
        app_features = app_feature.reshape(
            1, -1).expand(B, app_feature.shape[-1])
        xyz_samp = xyz.reshape(1, -1).expand(B, xyz.shape[-1])

        ele_grid, azi_grid = torch.meshgrid(
            torch.linspace(-np.pi/2, np.pi/2, res, dtype=torch.float32),
            torch.linspace(-np.pi, np.pi, 2*res, dtype=torch.float32), indexing='ij')
        # each col of x ranges from -pi/2 to pi/2
        # each row of y ranges from -pi to pi
        ang_vecs = torch.stack([
            torch.cos(ele_grid) * torch.cos(azi_grid),
            torch.cos(ele_grid) * torch.sin(azi_grid),
            -torch.sin(ele_grid),
        ], dim=-1).to(self.device)

        color, tint, matprop = self.diffuse_module(xyz_samp, ang_vecs.reshape(-1, 3), app_features)
        if self.ref_module is not None:
        # roughness = 1/np.pi*torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            roughness = matprop['roughness'] if roughness is None else roughness * torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            viewdotnorm = torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            envmap = (tint.reshape(-1, 3) * self.ref_module(xyz_samp, staticdir, app_features, refdirs=ang_vecs.reshape(
                -1, 3), roughness=roughness, viewdotnorm=viewdotnorm)).reshape(res, 2*res, 3)
        else:
            envmap = torch.zeros(res, 2*res, 3)
        color = (color).reshape(res, 2*res, 3)/2
        
        return self.tonemap(envmap).clamp(0, 1), self.tonemap(color).clamp(0, 1)

    def at_infinity(self, xyz_sampled, max_dist=10):
        margin = 1 - 1/max_dist/2
        at_infinity = torch.linalg.norm(
            xyz_sampled, dim=-1, ord=torch.inf).abs() >= margin
        return at_infinity

    def calculate_normals(self, xyz):
        with torch.enable_grad():
            xyz_g = xyz.clone()
            xyz_g.requires_grad = True

            # compute sigma
            xyz_g_normed = self.rf.normalize_coord(xyz_g)
            sigma_feature = self.rf.compute_densityfeature(xyz_g_normed)
            validsigma = self.feature2density(sigma_feature)

            # compute normal
            grad_outputs = torch.ones_like(validsigma)
            # TODO REMOVE
            g = grad(validsigma, xyz_g, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)
            # g = grad(validsigma, xyz_g, grad_outputs=grad_outputs, create_graph=False, allow_unused=True)
            norms = -g[0][:, :3]
            norms = norms / (torch.linalg.norm(norms, dim=-1, keepdim=True) + 1e-8)
            return norms

    def render_just_bg(self, rays_chunk, roughness, white_bg=True):
        if rays_chunk.shape[0] == 0:
            return torch.empty((0, 3), device=rays_chunk.device)
        viewdirs = rays_chunk[:, 3:6]
        bg = self.bg_module(viewdirs[:, :], roughness)
        return bg.reshape(-1, 3)

    def forward(self, rays_chunk, focal,
                recur=0, init_refraction_index=torch.tensor(1.0),
                override_near=None, output_alpha=None, white_bg=True,
                is_train=False, ndc_ray=False, N_samples=-1, tonemap=True):
        # rays_chunk: (N, (origin, viewdir, ray_up))

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
                rays_chunk[:, :3], viewdirs, focal, is_train=is_train, N_samples=N_samples, override_near=override_near)
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        # xyz_sampled_shape: (N, N_samples, 3+1)
        # z_vals.shape: (N, N_samples)
        # ray_valid.shape: (N, N_samples)
        xyz_sampled_shape = xyz_sampled[:, :, :3].shape

        xyz_normed = self.rf.normalize_coord(xyz_sampled)

        device = xyz_sampled.device

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled_shape)
        # rays_up = rays_chunk[:, 6:9]
        # rays_up = rays_up.view(-1, 1, 3).expand(xyz_sampled_shape)
        B = xyz_sampled.shape[0]
        n_samples = xyz_sampled_shape[1]

        ior_i_full = init_refraction_index.to(device).reshape(-1, 1).expand(xyz_sampled_shape[:-1])
        flip = ior_i_full > self.min_refraction

        # sample alphas and cull samples from the ray
        alpha_mask = torch.zeros(xyz_sampled_shape[:-1], device=device, dtype=bool)
        if self.alphaMask is not None and self.enable_alpha_mask and not flip.any():
            alpha_mask[ray_valid] = self.alphaMask.sample_alpha(
                xyz_sampled[ray_valid], contract_space=self.rf.contract_space)

            # T = torch.cumprod(torch.cat([
            #     torch.ones(alphas.shape[0], 1, device=alphas.device),
            #     1. - alphas + 1e-10
            # ], dim=-1), dim=-1)[:, :-1]
            # ray_invalid = ~ray_valid
            # ray_invalid |= (~alpha_mask)
            ray_valid ^= alpha_mask

        # sigma.shape: (N, N_samples)
        sigma = torch.zeros(xyz_sampled_shape[:-1], device=device)
        world_normal = torch.zeros(xyz_sampled_shape, device=device)
        rgb = torch.zeros((*xyz_sampled_shape[:2], 3), device=device)
        p_world_normal = torch.zeros(xyz_sampled_shape, device=device)

        if ray_valid.any():
            sigma_feature = self.rf.compute_densityfeature(xyz_normed[ray_valid])
            #  sigma_feature, world_normal[ray_valid] = self.rf.compute_density_norm(xyz_normed[ray_valid], self.feature2density)
            #  _, world_normal[ray_valid] = self.rf.compute_density_norm(xyz_normed[ray_valid], self.feature2density)
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

            #  with torch.enable_grad():
            #      xyz_g = xyz_sampled[ray_valid].clone()
            #      xyz_g.requires_grad = True
            #
            #      # compute sigma
            #      xyz_g_normed = self.rf.normalize_coord(xyz_g)
            #      sigma_feature = self.rf.compute_densityfeature(xyz_g_normed)
            #      validsigma = self.feature2density(sigma_feature)
            #      sigma[ray_valid] = validsigma
            #
            #      # compute normal
            #      grad_outputs = torch.ones_like(validsigma)
            #      surf_grad = grad(validsigma, xyz_g, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0][:, :3]
            #      surf_grad = surf_grad / (torch.norm(surf_grad, dim=1, keepdim=True)+1e-8)
            #
            #      world_normal[ray_valid] = surf_grad


        if self.rf.contract_space and self.infinity_border:
            at_infinity = self.at_infinity(xyz_normed)
            sigma[at_infinity] = 100

        # weight: [N_rays, N_samples]
        alpha, weight, bg_weight = raw2alpha(sigma, flip, dists * self.distance_scale)

        # ic(weight.sum(dim=1).mean(), override_near)
        # if override_near is not None:
        #     ic(weight.sum(dim=1).mean())
        #     ic(torch.sum(weight * z_vals, 1).mean())
        #     alpha, weight, bg_weight = raw2alpha(sigma, flip & False, dists * self.distance_scale)
        #     ic(torch.sum(weight * z_vals, 1).mean())
        #     ic(weight.sum(dim=1).mean())

        # if white_bg:
        #     # floater_loss = -torch.einsum('...j,...k->...', weight.reshape(B, -1), weight.reshape(B, -1)).mean()
        #     full_weight = weight
        # else:
        # weight[xyz_normed[..., 2] > 0.2] = 0
        full_weight = torch.cat([weight, bg_weight], dim=1)

        S = torch.linspace(0, 1, n_samples+1, device=device).reshape(-1, 1)
        fweight = (S - S.T).abs()

        floater_loss_1 = torch.einsum('bj,bk,jk', full_weight.reshape(B, -1), full_weight.reshape(B, -1), fweight).clip(min=self.max_floater_loss)
        floater_loss_2 = (full_weight**2).sum(dim=1).sum()/3/n_samples

        # this one consumes too much memory
        # floater_loss_1 = torch.einsum('bj,bk,jk->b', full_weight.reshape(B, -1), full_weight.reshape(B, -1), fweight).clip(min=self.max_floater_loss).sum()
        # ic(floater_loss_1, floater_loss_2)
        floater_loss = (floater_loss_1 + floater_loss_2)#.clip(min=self.max_floater_loss)

        # app stands for appearance
        app_mask = (weight > self.rayMarch_weight_thres)

        # debug = torch.zeros((B, n_samples, 3), dtype=torch.short, device=device)
        debug = torch.zeros((B, n_samples, 3), dtype=torch.float, device=device, requires_grad=False)
        recur_depth = z_vals.clone()
        depth_map = torch.sum(weight * recur_depth, 1)
        acc_map = bg_weight #torch.sum(weight, 1)
        depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]
        bounce_count = 0

        if app_mask.any():
            #  Compute normals for app mask

            norms = self.calculate_normals(xyz_sampled[app_mask])
            world_normal[app_mask] = norms
            # pred norms is initialized to world norms to set loss to zero for align_world_loss when prediction is none
            p_world_normal[app_mask] = norms.detach()

            app_features = self.rf.compute_appfeature(xyz_normed[app_mask])

            # get base color of the point
            diffuse, tint, matprop = self.diffuse_module(
                xyz_normed[app_mask], viewdirs[app_mask], app_features)
            # diffuse = diffuse.type(rgb.dtype)

            noise_app_features = (app_features + torch.randn_like(app_features) * self.appdim_noise_std)

            # interpolate between the predicted and world normals
            if self.normal_module is not None:
                p_world_normal[app_mask] = self.normal_module(xyz_normed[app_mask], app_features)
                l = self.l# if is_train else 1
                v_world_normal = ((1-l)*p_world_normal + l*world_normal)
                v_world_normal = v_world_normal / (v_world_normal.norm(dim=-1, keepdim=True) + 1e-20)
            else:
                v_world_normal = world_normal

            # calculate reflected ray direction
            V = -viewdirs[app_mask]
            L = v_world_normal[app_mask]
            refdirs = 2 * (V * L).sum(-1, keepdim=True) * L - V

            reflect_rgb = torch.zeros_like(diffuse)
            roughness = matprop['roughness']
            if recur >= self.max_recurs and self.ref_module is None:
                ratio_diffuse = 1
                ratio_reflected = 0
            elif self.ref_module is not None and recur >= self.max_recurs:
                viewdotnorm = (viewdirs[app_mask]*L).sum(dim=-1, keepdim=True)
                ref_col = self.ref_module(
                    xyz_normed[app_mask], viewdirs[app_mask],
                    noise_app_features, refdirs=refdirs,
                    roughness=roughness, viewdotnorm=viewdotnorm)
                reflect_rgb = tint * ref_col
                debug[app_mask] += ref_col / (ref_col + 1)
            else:
                num_roughness_rays = self.roughness_rays // 2 if recur > 0 else self.roughness_rays
                # num_roughness_rays = self.roughness_rays# if is_train else 100
                # compute which rays to reflect
                # TODO REMOVE
                # bounce_mask, full_bounce_mask, inv_full_bounce_mask = select_top_n_app_mask(
                #         app_mask, weight, ratio_reflected, self.max_bounce_rays,
                #         self.specularity_threshold, self.bounce_min_weight)
                ratio_diffuse = matprop['ratio_diffuse']
                ratio_reflected = 1 - ratio_diffuse
                bounce_mask, full_bounce_mask, inv_full_bounce_mask = self.selector(
                        app_mask, weight.detach(), 1-roughness.detach())
                # if the bounce is not calculated, set the ratio to 0 to make sure we don't get black spots
                # if not bounce_mask.all() and not is_train:
                #     ratio_diffuse[~bounce_mask] += ratio_reflected[~bounce_mask]
                #     ratio_reflected[~bounce_mask] = 0

                if bounce_mask.sum() > 0:
                    # decide how many bounces to calculate
                    brefdirs = refdirs[bounce_mask].reshape(-1, 1, 3)
                    # add noise to simulate roughness
                    N = brefdirs.shape[0]
                    outward = L[bounce_mask]
                    # ray_noise = self.roughness2noisestd(roughness[bounce_mask].reshape(-1, 1, 1)) * torch.normal(0, 1, (N, num_roughness_rays, 3), device=device)
                    # diffuse_noise = ray_noise / (torch.linalg.norm(ray_noise, dim=-1, keepdim=True)+1e-8)
                    # noise_rays = self.sampler.sample(num_roughness_rays, V[bounce_mask].detach(), outward.detach(), roughness[bounce_mask].detach())
                    # noise_rays, mipval = self.sampler.sample(num_roughness_rays, brefdirs.detach(), V[bounce_mask], outward.detach(), roughness[bounce_mask])
                    noise_rays, mipval = self.sampler.sample(num_roughness_rays, brefdirs, V[bounce_mask], outward, roughness[bounce_mask])
                    bounce_rays = torch.cat([
                        xyz_sampled[full_bounce_mask][..., :3].reshape(-1, 1, 3).expand(noise_rays.shape),
                        noise_rays,
                        # rays_up[full_bounce_mask].reshape(-1, 1, 3).expand(noise_rays.shape)
                    ], dim=-1)
                    D = bounce_rays.shape[-1]

                    # ray_mask = (torch.arange(num_roughness_rays, device=device).reshape(1, -1, 1) < (roughness[bounce_mask] * num_roughness_rays).clip(min=1).reshape(-1, 1, 1))
                    # ray_mask[:, 1:] &= ((noise_rays * brefdirs).sum(dim=-1, keepdim=True) < 0.99999)[:, 1:]
                    ray_mask = ((noise_rays * brefdirs).sum(dim=-1, keepdim=True) < 1-5e-5)
                    ray_mask[:, 0] = True
                    # ray_mask = torch.sigmoid(torch.arange(num_roughness_rays, device=device).reshape(1, -1, 1) - (roughness[bounce_mask] * num_roughness_rays).clip(min=1).reshape(-1, 1, 1))

                    if recur == 0 and self.world_bounces > 0:
                        incoming_light = torch.empty((bounce_rays.shape[0], bounce_rays.shape[1], 3), device=device)
                        with torch.no_grad():
                            reflect_data = self(bounce_rays[:, :self.world_bounces, :].reshape(-1, D), focal, recur=recur+1, white_bg=False,
                                                override_near=0.15, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples, tonemap=False)


                        incoming_light[:, :self.world_bounces, :] = reflect_data['rgb_map'].reshape(-1, self.world_bounces, 3)
                        mipval = mipval.reshape(-1, num_roughness_rays)
                        # apply ray mask
                        incoming_light[:, self.world_bounces:, :] = self.render_just_bg(
                                bounce_rays[:, self.world_bounces:, :].reshape(-1, D),
                                mipval[:, self.world_bounces:].reshape(-1, 1)
                            ).reshape(-1, num_roughness_rays-self.world_bounces, 3)
                    else:
                        incoming_light = self.render_just_bg(bounce_rays.reshape(-1, D), mipval).reshape(-1, num_roughness_rays, 3)
                    # miplevel = self.bg_module.sa2mip(mipval)
                    # debug[full_bounce_mask][..., 0] += miplevel.mean(dim=1) / (self.bg_module.max_mip-1)
                    

                    incoming_light = ray_mask * incoming_light + (~ray_mask) * incoming_light[:, 0:1, :]
                    ray_mask = ray_mask | True

                    tinted_ref_rgb = self.brdf(incoming_light, V[bounce_mask], bounce_rays[..., 3:6], outward.reshape(-1, 1, 3), noise_app_features[bounce_mask], matprop, bounce_mask, ray_mask)
                    s = incoming_light.mean(dim=1)
                    # s = incoming_light[:, 0]
                    debug[full_bounce_mask] += s / (s+1)
                    # debug[full_bounce_mask] += bounce_rays[:, 0, 3:6]/2 + 0.5
                    # reflect_rgb[bounce_mask] = tint[bounce_mask] * tinted_ref_rgb
                    reflect_rgb[bounce_mask] = tinted_ref_rgb

                    # m = full_bounce_mask.sum(dim=1) > 0
                    # LOGGER.log_rays(rays_chunk[m].reshape(-1, D), recur, dict(depth_map=depth_map.detach()[m]))
                    # LOGGER.log_rays(bounce_rays.reshape(-1, D), recur+1, reflect_data)
                bounce_count = bounce_mask.sum()

                if inv_full_bounce_mask.any():
                    if self.ref_module is not None:
                        # compute other reflections using ref module
                        viewdotnorm = (viewdirs[inv_full_bounce_mask]*L[~bounce_mask]).sum(dim=-1, keepdim=True)
                        ref_col = self.ref_module(
                            xyz_normed[inv_full_bounce_mask], viewdirs[inv_full_bounce_mask],
                            noise_app_features[~bounce_mask], refdirs=refdirs[~bounce_mask],
                            roughness=roughness[~bounce_mask], viewdotnorm=viewdotnorm)
                        reflect_rgb[~bounce_mask] = tint[~bounce_mask] * ref_col
                        # debug[inv_full_bounce_mask] += ref_col
                    else:
                        reflect_rgb[~bounce_mask] = tint[~bounce_mask]*matprop['ambient'][~bounce_mask]
                        # debug[inv_full_bounce_mask] += tint[~bounce_mask]*matprop['ambient'][~bounce_mask]

                bounce_count = bounce_mask.sum()
            # this is a modified rendering equation where the emissive light and light under the integral is all multiplied by the base color
            # in addition, the light is interpolated between emissive and reflective
            reflectivity = matprop['reflectivity']
            # rgb[app_mask] = tint * ((1-reflectivity)*matprop['ambient'] + reflectivity * reflect_rgb)
            # rgb[app_mask] = reflect_rgb + matprop['diffuse']
            rgb[app_mask] = reflect_rgb# + matprop['diffuse']
            # rgb[app_mask] = tint * reflectivity * reflect_rgb + (1-reflectivity)*matprop['diffuse']
            # rgb[app_mask] = tint * (ambient + reflectivity * reflect_rgb)

            align_world_loss = (1-(p_world_normal * world_normal).sum(dim=-1).clamp(max=self.max_normal_similarity))
            # align_world_loss = torch.linalg.norm(p_world_normal - world_normal, dim=-1)
            normal_loss = (weight * align_world_loss).sum(dim=-1).mean()
            tint_brightness = tint.mean(dim=-1)
        else:
            ratio_diffuse = torch.tensor(0.0)
            diffuse = torch.tensor(0.0)
            tint_brightness = torch.tensor(0.5)
            v_world_normal = world_normal
            roughness = torch.tensor(0.0)
            normal_loss = torch.tensor(0.0)
            reflectivity = torch.tensor(0.0)
        
        # viewdirs point inward. -viewdirs aligns with p_world_normal. So we want it below 0
        backwards_rays_loss = torch.matmul(viewdirs.reshape(-1, 1, 3), p_world_normal.reshape(-1, 3, 1)).reshape(app_mask.shape).clamp(min=0)**2
        backwards_rays_loss = (weight * backwards_rays_loss).sum(dim=1).mean()

        # calculate depth

        # shadow_map = torch.sum(weight * shadows, 1)
        # (N, bundle_size, bundle_size)
        acc_map = torch.sum(weight, 1)
        with torch.no_grad():
            depth_map = torch.sum(weight * recur_depth, 1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

            # view dependent normal map
            # N, 3, 3
            # row_basis = -torch.stack([
            #     -torch.linalg.cross(viewdirs[:, 0], rays_up[:, 0], dim=-1),
            #     viewdirs[:, 0],
            #     rays_up[:, 0],
            # ], dim=1)
            # p_world_normal_map = torch.sum(weight[..., None] * p_world_normal, 1)
            # p_world_normal_map = p_world_normal_map / \
            #     (torch.norm(p_world_normal_map, dim=-1, keepdim=True)+1e-8)
            # d_world_normal_map = torch.sum(weight[..., None] * world_normal, 1)
            # d_world_normal_map = d_world_normal_map / (torch.linalg.norm(d_world_normal_map, dim=-1, keepdim=True)+1e-8)
            v_world_normal_map = torch.sum(weight[..., None] * v_world_normal, 1)
            v_world_normal_map = v_world_normal_map / (torch.linalg.norm(v_world_normal_map, dim=-1, keepdim=True)+1e-8)
            # d_normal_map = torch.matmul(row_basis, d_world_normal_map.unsqueeze(-1)).squeeze(-1)
            # p_normal_map = torch.matmul(
            #     row_basis, p_world_normal_map.unsqueeze(-1)).squeeze(-1)
            # v_normal_map = torch.matmul(row_basis, v_world_normal_map.unsqueeze(-1)).squeeze(-1)
            # v_normal_map = v_normal_map / (torch.linalg.norm(d_normal_map, dim=-1, keepdim=True)+1e-8)
            v_world_normal_map = acc_map[..., None] * v_world_normal_map + (1 - acc_map[..., None])

            inds = ((weight * (alpha < self.alphaMask_thres)).max(dim=1).indices).clip(min=0)
            termination_xyz = xyz_sampled[range(xyz_sampled_shape[0]), inds]

            # collect statistics about the surface
            # surface width in voxels
            surface_width = (torch.arange(weight.shape[1], device=device)[None, :] * weight).std(dim=1)
            weight_slice = weight[torch.where(acc_map > 0.5)[0]].reshape(1, -1)
            # TODO REMOVE
            LOGGER.log_norms_n_rays(xyz_sampled, v_world_normal, weight)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if tonemap:
            rgb_map = self.tonemap(rgb_map.clip(min=0), noclip=True)

        if self.bg_module is not None and not white_bg:
            bg_roughness = torch.zeros(B, 1, device=device)
            bg = self.bg_module(viewdirs[:, 0, :], bg_roughness)
            rgb_map = rgb_map + \
                (1 - acc_map[..., None]) * self.tonemap(bg.reshape(-1, 3), noclip=True)
        else:
            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1 - acc_map[..., None])

        debug_map = (weight[..., None]*debug).sum(dim=1)

        return dict(
            rgb_map=rgb_map,
            depth_map=depth_map.detach().cpu(),
            debug_map=debug_map.detach().cpu(),
            normal_map=v_world_normal_map.detach().cpu(),
            weight_slice=weight_slice,
            recur=recur,
            acc_map=acc_map.detach().cpu(),
            roughness=roughness.mean(),
            diffuse_reg=roughness.mean() - reflectivity.mean() + diffuse.mean(),# + ((tint_brightness-0.5)**2).mean(),
            normal_loss=normal_loss,
            backwards_rays_loss=backwards_rays_loss,
            termination_xyz=termination_xyz.detach().cpu(),
            floater_loss=floater_loss,
            surf_width=surface_width,
            color_count=app_mask.detach().sum(),
            bounce_count=bounce_count,
        )
