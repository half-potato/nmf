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
from modules import distortion_loss, row_mask_sum
from mutils import normalize

LOGGER = Logger(enable=False)
FIXED_SPHERE = False
FIXED_RETRO = False

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp((-sigma*dist).clip(min=-1e10, max=0))

    # T is the term that integrates the alpha backwards to prevent occluded objects from influencing things
    # multiply in exponential space to take exponential of integral
    T = torch.cumprod(torch.cat([
        torch.ones(alpha.shape[0], 1, device=alpha.device),
        1. - alpha + 1e-10
    ], dim=-1), dim=-1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]

class TensorNeRF(torch.nn.Module):
    def __init__(self, rf, aabb, near_far,
                 diffuse_module, sampler, brdf_sampler=None, brdf=None, tonemap=None, normal_module=None, ref_module=None, bg_module=None,
                 visibility_module=None,
                 alphaMask=None, specularity_threshold=0.005, max_recurs=0,
                 max_normal_similarity=1, infinity_border=False, min_refraction=1.1, enable_refraction=True,
                 alphaMask_thres=0.001, rayMarch_weight_thres=0.0001,
                 max_bounce_rays=4000, roughness_rays=3, bounce_min_weight=0.001, appdim_noise_std=0.0,
                 world_bounces=0, selector=None, cold_start_bg_iters = 0,
                 update_sampler_list=[5000], max_floater_loss=6, **kwargs):
        super(TensorNeRF, self).__init__()
        self.rf = rf(aabb=aabb)
        self.ref_module = ref_module(in_channels=self.rf.app_dim) if ref_module is not None else None
        self.normal_module = normal_module(in_channels=self.rf.app_dim) if normal_module is not None else None
        bound = aabb.abs().max()
        self.diffuse_module = diffuse_module(in_channels=self.rf.app_dim)
        al = self.diffuse_module.allocation
        self.normal_module = normal_module(in_channels=self.rf.app_dim-al) if normal_module is not None else None
        al += self.normal_module.allocation
        self.ref_module = ref_module(in_channels=self.rf.app_dim-al) if ref_module is not None else None
        self.brdf = brdf(in_channels=self.rf.app_dim-al) if brdf is not None else None
        self.brdf_sampler = brdf_sampler if brdf_sampler is None else brdf_sampler(num_samples=roughness_rays)
        self.visibility_module = visibility_module(in_channels=self.rf.app_dim-al, bound=bound) if visibility_module is not None else None
        self.sampler = sampler(near_far=near_far, aabb=aabb)
        ic(self.sampler)
        self.selector = selector
        self.bg_module = bg_module
        if tonemap is None:
            self.tonemap = SRGBTonemap()
        else:
            self.tonemap = tonemap

        self.world_bounces = world_bounces
        self.alphaMask = alphaMask
        self.infinity_border = infinity_border
        self.max_floater_loss = max_floater_loss
        self.alphaMask_thres = alphaMask_thres
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.appdim_noise_std = appdim_noise_std
        self.update_sampler_list = update_sampler_list

        self.bounce_min_weight = bounce_min_weight
        self.min_refraction = min_refraction
        self.enable_refraction = enable_refraction
        self.roughness_rays = roughness_rays
        self.max_recurs = max_recurs
        self.specularity_threshold = specularity_threshold
        self.max_bounce_rays = max_bounce_rays

        self.cold_start_bg_iters = cold_start_bg_iters
        self.detach_bg = True


        f_blur = torch.tensor([1, 2, 1]) / 4
        f_edge = torch.tensor([-1, 0, 1]) / 2
        self.register_buffer('f_blur', f_blur)
        self.register_buffer('f_edge', f_edge)

        self.max_normal_similarity = max_normal_similarity
        self.l = 0
        # self.sampler.update(self.rf, init=True)
        # self.sampler2.update(self.rf, init=True)

    def get_device(self):
        return self.rf.units.device

    def get_optparam_groups(self, lr_bg=0.025, lr_scale=1):
        grad_vars = []
        grad_vars += self.rf.get_optparam_groups()
        if isinstance(self.normal_module, torch.nn.Module):
            grad_vars += [{'params': self.normal_module.parameters(),
                           'lr': self.normal_module.lr*lr_scale}]
        if self.ref_module is not None:
            grad_vars += [{'params': list(self.ref_module.parameters()), 'lr': lr_scale*self.ref_module.lr}]
        if isinstance(self.diffuse_module, torch.nn.Module):
            grad_vars += [{'params': self.diffuse_module.parameters(),
                           'lr': self.diffuse_module.lr}]
        if isinstance(self.brdf, torch.nn.Module):
            grad_vars += [{'params': self.brdf.parameters(),
                           'lr': self.brdf.lr}]
        if isinstance(self.bg_module, torch.nn.Module):
            grad_vars += self.bg_module.get_optparam_groups()
        return grad_vars

    def save(self, path, config):
        print(f"Saving nerf to {path}")
        if self.bg_module is not None:
            config['bg_module']['bg_resolution'] = self.bg_module.bg_resolution
        ckpt = {'config': config, 'state_dict': self.state_dict()}
        # if self.alphaMask is not None:
        #     alpha_volume = self.alphaMask.alpha_volume.cpu()
        #     ckpt.update({'alphaMask': alpha_volume})
        #     #  alpha_volume = self.alphaMask.alpha_volume.cpu().numpy()
        #     #  ckpt.update({'alphaMask.shape': alpha_volume.shape})
        #     #  ckpt.update(
        #     #      {'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
        #     ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    @staticmethod
    def load(ckpt, config=None, near_far=None, **kwargs):
        config = ckpt['config'] if config is None else config
        aabb = ckpt['state_dict']['rf.aabb']
        near_far = near_far if near_far is not None else [1, 6]
        rf = hydra.utils.instantiate(config)(aabb=aabb, near_far=near_far)
        # if 'alphaMask.aabb' in ckpt.keys():
        #     #  length = np.prod(ckpt['alphaMask.shape'])
        #     #  alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[
        #     #                                  :length].reshape(ckpt['alphaMask.shape'])).float()
        #     alpha_volume = ckpt['alphaMask']
        #     rf.alphaMask = utils.AlphaGridMask(
        #         ckpt['alphaMask.aabb'], alpha_volume)
        rf.load_state_dict(ckpt['state_dict'], **kwargs)
        return rf

    # @torch.no_grad()
    # def filtering_rays(self, all_rays, all_rgbs, focal, N_samples=256, chunk=10240*5, bbox_only=False):
    #     print('========> filtering rays ...')
    #     tt = time.time()
    #
    #     N = torch.tensor(all_rays.shape[:-1]).prod()
    #
    #     mask_filtered = []
    #     idx_chunks = torch.split(torch.arange(N), chunk)
    #     for idx_chunk in idx_chunks:
    #         rays_chunk = all_rays[idx_chunk].to(self.get_device())
    #
    #         rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
    #         if bbox_only:
    #             vec = torch.where(
    #                 rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
    #             rate_a = (self.rf.aabb[1].to(rays_o) - rays_o) / vec
    #             rate_b = (self.rf.aabb[0].to(rays_o) - rays_o) / vec
    #             # .clamp(min=near, max=far)
    #             t_min = torch.minimum(rate_a, rate_b).amax(-1)
    #             # .clamp(min=near, max=far)
    #             t_max = torch.maximum(rate_a, rate_b).amin(-1)
    #             mask_inbbox = t_max > t_min
    #
    #         else:
    #             xyz_sampled, _, _, _ = self.sample_ray(
    #                 rays_o, rays_d, focal, N_samples=N_samples, is_train=False)
    #             # Issue: calculate size
    #             mask_inbbox = self.alphaMask.sample_alpha(
    #                     xyz_sampled).reshape(xyz_sampled.shape[:-1]).any(-1)
    #
    #         mask_filtered.append(mask_inbbox.cpu())
    #
    #     mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
    #
    #     print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
    #     return all_rays[mask_filtered], all_rgbs[mask_filtered], mask_filtered

    def sample_occupied(self):
        samps = torch.rand((10000, 4), device=self.get_device())*2 - 1
        validsigma = self.rf.compute_densityfeature(samps).squeeze()
        mask = validsigma > validsigma.mean()
        inds, = torch.where(mask)
        ind = random.randint(0, len(inds))
        xyz = samps[inds[ind]]
        return xyz

    def check_schedule(self, iter):
        require_reassignment = False
        require_reassignment |= self.rf.check_schedule(iter)
        require_reassignment |= self.sampler.check_schedule(iter, self.rf)
        if require_reassignment:
            self.sampler.update(self.rf, init=True)
        if iter > self.cold_start_bg_iters:
            self.detach_bg = False
        return require_reassignment

    def render_env_sparse(self, ray_origins, env_dirs, roughness: float):
        B, M = env_dirs.shape[:2]
        ray_origins = torch.cat([ray_origins, roughness*torch.ones((B, 1), device=self.get_device())], dim=-1)
        norm_ray_origins = self.rf.normalize_coord(ray_origins)
        app_features = self.rf.compute_appfeature(norm_ray_origins)
        app_features = app_features.reshape(B, 1, -1).expand(B, M, -1)
        norm_ray_origins = norm_ray_origins.reshape(B, 1, -1).expand(B, M, -1)
        roughness = torch.tensor(roughness, device=ray_origins.device)
        staticdir = torch.zeros((B*M, 3), device=self.get_device())
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

        device = xyz.device
        app_feature = self.rf.compute_appfeature(xyz.reshape(1, -1))
        B = 2*res*res
        staticdir = torch.zeros((B, 3), device=device)
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
        ], dim=-1).to(device)

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
            validsigma = self.rf.compute_densityfeature(xyz_g_normed)

            # compute normal
            grad_outputs = torch.ones_like(validsigma)
            g = grad(validsigma, xyz_g, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)
            norms = normalize(-g[0][:, :3])
            return norms

    def init_vis_module(self, S=4, G=8):
        device = self.get_device()
        _theta = torch.linspace(-np.pi/2, np.pi/2, G//2, device=device)
        _phi = torch.linspace(-np.pi, np.pi, G, device=device)
        _theta += torch.rand(1, device=device)
        _phi += torch.rand(1, device=device)
        theta, phi = torch.meshgrid(_theta, _phi, indexing='ij')
        pviewdirs = torch.stack([
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            -torch.sin(theta),
        ], dim=-1).reshape(-1, 1, 3)

        X = torch.linspace(0, 1, self.visibility_module.grid_size, device=device).split(S)
        Y = torch.linspace(0, 1, self.visibility_module.grid_size, device=device).split(S)
        Z = torch.linspace(0, 1, self.visibility_module.grid_size, device=device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    # construct points
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    samples = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 1)
                    samples = (samples + (torch.rand_like(samples)) / (self.visibility_module.grid_size-1)).clip(0, 1)
                    porigins = self.rf.aabb[0] * (1-samples) + self.rf.aabb[1] * samples

                    origins = porigins.reshape(1, -1, 3).expand(pviewdirs.shape[0], -1, -1).reshape(-1, 3)
                    viewdirs = pviewdirs.expand(-1, porigins.shape[0], 3).reshape(-1, 3)
                    
                    rays = torch.cat([ origins, viewdirs ], dim=-1)

                    # pass rays to sampler to compute expected termination
                    xyz_sampled, ray_valid, max_samps, z_vals, dists, whole_valid = self.sampler.sample(
                        rays, 0.1, False, override_near=0.05, is_train=False, N_samples=-1)

                    # xyz_sampled: (M, 4) float. premasked valid sample points
                    # ray_valid: (b, N) bool. mask of which samples are valid
                    # max_samps = N
                    # z_vals: (b, N) float. distance along ray to sample
                    # dists: (b, N) float. distance between samples
                    # whole_valid: mask into origin rays_chunk of which B rays where able to be fully sampled.
                    # """
                    B = ray_valid.shape[0]
                    xyz_normed = self.rf.normalize_coord(xyz_sampled)
                    full_shape = (B, max_samps, 3)
                    sigma = torch.zeros(full_shape[:-1], device=device)

                    if ray_valid.any():
                        if self.rf.separate_appgrid:
                            psigma = self.rf.compute_densityfeature(xyz_normed)
                        else:
                            psigma, all_app_features = self.rf.compute_feature(xyz_normed)
                        sigma[ray_valid] = psigma

                    # weight: [N_rays, N_samples]
                    alpha, weight, bg_weight = raw2alpha(sigma, dists * self.rf.distance_scale)

                    # app_features = self.rf.compute_appfeature(norm_ray_origins)
                    termination = (z_vals * weight).median()
                    bgvisibility = bg_weight[..., 0] > 0.1
                    # """
                    # bgvisibility = ray_valid.sum(dim=1) > 0
                    normed_origins = self.rf.normalize_coord(origins)

                    self.visibility_module.update(normed_origins, viewdirs, bgvisibility)
        torch.cuda.empty_cache()
        return torch.tensor(0.0, device=device)

    def compute_visibility_loss(self, N, G=16):
        # generate random rays within aabb
        device = self.get_device()
        samples = torch.rand(N // 2, 3, device=device)
        # interpolate
        porigins = self.rf.aabb[0] * (1-samples) + self.rf.aabb[1] * samples
        _, _, xyz = self.sampler.sample_occupied(-1, N//2)
        porigins = torch.cat([porigins, xyz], dim=0)

        _theta = torch.linspace(-np.pi/2, np.pi/2, G//2, device=device)
        _phi = torch.linspace(-np.pi, np.pi, G, device=device)
        _theta += torch.rand(1, device=device)
        _phi += torch.rand(1, device=device)
        theta, phi = torch.meshgrid(_theta, _phi, indexing='ij')
        pviewdirs = torch.stack([
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            -torch.sin(theta),
        ], dim=-1).reshape(-1, 1, 3)
        origins = porigins.reshape(1, -1, 3).expand(pviewdirs.shape[0], -1, -1).reshape(-1, 3)
        viewdirs = pviewdirs.expand(-1, porigins.shape[0], 3).reshape(-1, 3)

        rays = torch.cat([origins, viewdirs], dim=-1)

        # pass rays to sampler to compute expected termination
        xyz_sampled, ray_valid, max_samps, z_vals, dists, whole_valid = self.sampler.sample(
            rays, 0.1, False, override_near=0.05, is_train=False, N_samples=-1)

        # xyz_sampled: (M, 4) float. premasked valid sample points
        # ray_valid: (b, N) bool. mask of which samples are valid
        # max_samps = N
        # z_vals: (b, N) float. distance along ray to sample
        # dists: (b, N) float. distance between samples
        # whole_valid: mask into origin rays_chunk of which B rays where able to be fully sampled.
        # """
        B = ray_valid.shape[0]
        xyz_normed = self.rf.normalize_coord(xyz_sampled)
        full_shape = (B, max_samps, 3)
        sigma = torch.zeros(full_shape[:-1], device=device)

        if ray_valid.any():
            if self.rf.separate_appgrid:
                psigma = self.rf.compute_densityfeature(xyz_normed)
            else:
                psigma, all_app_features = self.rf.compute_feature(xyz_normed)
            sigma[ray_valid] = psigma

        # weight: [N_rays, N_samples]
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.rf.distance_scale)
        # """

        # termination = (z_vals * weight).median()
        # bgvisibility = ray_valid.sum(dim=1) > 0
        bgvisibility = bg_weight[..., 0] > 0.1
        # ic(bgvisibility.sum() / bgvisibility.numel())

        # get value from MLP and compare to get loss
        norm_ray_origins = self.rf.normalize_coord(origins)
        # app_features = self.rf.compute_appfeature(norm_ray_origins)
        torch.cuda.empty_cache()
        return self.visibility_module.update(norm_ray_origins, viewdirs, bgvisibility)

    def render_just_bg(self, rays_chunk, roughness, white_bg=True):
        if rays_chunk.shape[0] == 0:
            return torch.empty((0, 3), device=rays_chunk.device)
        viewdirs = rays_chunk[:, 3:6]
        bg = self.bg_module(viewdirs[:, :], roughness)
        if self.detach_bg:
            bg = bg.detach()
        return bg.reshape(-1, 3)

    def forward(self, rays_chunk, focal, start_mipval=None,
                recur=0, override_near=None, output_alpha=None, white_bg=True,
                is_train=False, ndc_ray=False, N_samples=-1, tonemap=True):
        # rays_chunk: (N, (origin, viewdir, ray_up))
        output = {}

        # sample points
        device = rays_chunk.device

        xyz_sampled, ray_valid, max_samps, z_vals, dists, whole_valid = self.sampler.sample(
            rays_chunk, focal, ndc_ray, override_near=override_near, is_train=is_train, N_samples=N_samples)
        # xyz_sampled: (M, 4) float. premasked valid sample points
        # ray_valid: (b, N) bool. mask of which samples are valid
        # max_samps = N
        # z_vals: (b, N) float. distance along ray to sample
        # dists: (b, N) float. distance between samples
        # whole_valid: mask into origin rays_chunk of which B rays where able to be fully sampled.
        B = ray_valid.shape[0]

        xyz_normed = self.rf.normalize_coord(xyz_sampled)
        full_shape = (B, max_samps, 3)
        n_samples = full_shape[1]

        M = xyz_sampled.shape[0]
 
        device = xyz_sampled.device

        viewdirs = rays_chunk[whole_valid, 3:6].view(-1, 1, 3).expand(full_shape)
        # rays_up = rays_chunk[:, 6:9]
        # rays_up = rays_up.view(-1, 1, 3).expand(full_shape)
        n_samples = full_shape[1]

        # sigma.shape: (N, N_samples)
        sigma = torch.zeros(full_shape[:-1], device=device)

        world_normal = torch.zeros((M, 3), device=device)
        p_world_normal = torch.zeros((M, 3), device=device)

        all_app_features = None
        if FIXED_SPHERE:
            sigma[ray_valid] = torch.where(torch.linalg.norm(xyz_sampled, dim=-1) < 1, 99999999.0, 0.0)
        elif FIXED_RETRO:
            mask1 = torch.linalg.norm(xyz_sampled, dim=-1, ord=torch.inf) < 0.613
            mask2 = (xyz_sampled[..., 0] < 0) & (xyz_sampled[..., 1] > 0)
            sigma[ray_valid] = torch.where(mask1 & ~mask2, 99999999.0, 0.0)
        else:
            if ray_valid.any():
                if self.rf.separate_appgrid:
                    psigma = self.rf.compute_densityfeature(xyz_normed)
                else:
                    psigma, all_app_features = self.rf.compute_feature(xyz_normed)
                sigma[ray_valid] = psigma


        if self.rf.contract_space and self.infinity_border:
            at_infinity = self.at_infinity(xyz_normed)
            sigma[at_infinity] = 100

        # weight: [N_rays, N_samples]
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.rf.distance_scale)

        # app stands for appearance
        pweight = weight[ray_valid]
        app_mask = (weight > self.rayMarch_weight_thres)
        papp_mask = app_mask[ray_valid]

        # if self.visibility_module is not None:
        #     self.visibility_module.ray_update(xyz_normed, viewdirs[ray_valid], app_mask, ray_valid)

        debug = torch.zeros((B, n_samples, 3), dtype=torch.float, device=device, requires_grad=False)
        bounce_count = 0

        rgb = torch.zeros((*full_shape[:2], 3), device=device, dtype=weight.dtype)

        if app_mask.any():
            #  Compute normals for app mask

            # TODO REMOVE
            norms = self.calculate_normals(xyz_sampled[papp_mask])
            world_normal[papp_mask] = norms
            # pred norms is initialized to world norms to set loss to zero for align_world_loss when prediction is none
            p_world_normal[papp_mask] = norms.detach()

            app_xyz = xyz_normed[papp_mask]

            if all_app_features is None:
                app_features = self.rf.compute_appfeature(app_xyz)
            else:
                app_features = all_app_features[papp_mask]
                # _, app_features = self.rf.compute_feature(app_xyz)

            noise_app_features = (app_features + torch.randn_like(app_features) * self.appdim_noise_std)

            # get base color of the point
            diffuse, tint, matprop = self.diffuse_module(
                app_xyz, viewdirs[app_mask], app_features)

            app_features = app_features[..., self.diffuse_module.allocation:]
            noise_app_features = noise_app_features[..., self.diffuse_module.allocation:]
            # diffuse = diffuse.type(rgb.dtype)

            # interpolate between the predicted and world normals
            if self.normal_module is not None:
                p_world_normal[papp_mask] = self.normal_module(app_xyz, app_features)
                l = self.l
                v_world_normal = normalize((1-l)*p_world_normal + l*world_normal)
                # TODO REMOVE
                if FIXED_SPHERE:
                    v_world_normal = normalize(xyz_sampled[..., :3])
                elif FIXED_RETRO:
                    xyz = xyz_sampled[..., :3]
                    eps = 3e-2
                    sqnorm = normalize(xyz, ord=torch.inf)
                    sqnorm = torch.sign(sqnorm) * (sqnorm.abs() > 1-eps).float()
                    left = torch.tensor([-1.0, 0.0, 0.0], device=device).reshape(1, 3).expand_as(xyz)
                    up = torch.tensor([0.0, 1.0, 0.0], device=device).reshape(1, 3).expand_as(xyz)
                    # corner_norms = torch.where(xyz[..., 0:1] - xyz[..., 1:2] > 0, left, up)
                    corner_norms = torch.where(xyz[..., 1:2] + xyz[..., 0:1] > 0, left, up)
                    # v_world_normal = torch.where(mask, corner_norms, sqnorm)
                    v_world_normal = sqnorm
                    mask = (xyz[..., 0] < eps) & (xyz[..., 1] > -eps)
                    v_world_normal[mask] = corner_norms[mask]
                    # v_world_normal = normalize(corner_norms)

                # v_world_normal = xyz_sampled[..., :3] / (xyz_sampled[..., :3].norm(dim=-1, keepdim=True) + 1e-8)
                # world_normal = xyz_sampled[..., :3] / (xyz_sampled[..., :3].norm(dim=-1, keepdim=True) + 1e-8)
            else:
                v_world_normal = world_normal

            app_features = app_features[..., self.normal_module.allocation:]
            noise_app_features = noise_app_features[..., self.normal_module.allocation:]

            # calculate reflected ray direction
            V = -viewdirs[app_mask]
            L = v_world_normal[papp_mask]
            VdotL = (V * L).sum(-1, keepdim=True)
            refdirs = 2 * VdotL * L - V


            reflect_rgb = torch.zeros_like(diffuse)
            roughness = matprop['roughness'].squeeze(-1)
            # roughness = 1e-2*torch.ones_like(roughness)
            # roughness = torch.where((xyz_sampled[..., 0].abs() < 0.15) | (xyz_sampled[..., 1].abs() < 0.15), 0.30, 0.15)[papp_mask]
            if recur >= self.max_recurs and self.ref_module is None:
                pass
            # TODO REMOVE
            elif self.ref_module is not None and recur >= self.max_recurs and not is_train:
                viewdotnorm = (viewdirs[app_mask]*L).sum(dim=-1, keepdim=True)
                ref_col = self.ref_module(
                    app_xyz, viewdirs[app_mask],
                    noise_app_features, refdirs=refdirs,
                    roughness=roughness, viewdotnorm=viewdotnorm)
                reflect_rgb = tint * ref_col
                debug[app_mask] += ref_col / (ref_col + 1)
            else:
                num_roughness_rays = self.roughness_rays // 2 if recur > 0 else self.roughness_rays
                # compute which rays to reflect
                bounce_mask, full_bounce_mask, inv_full_bounce_mask, ray_mask = self.selector(
                        app_mask, weight.detach(), VdotL, 1-roughness.detach(), num_roughness_rays)
                # bounce mask says which of the sampled points has greater than 0 rays
                # ray mask assumes that each of the bounce mask points has num_roughness_rays and masks out rays from each point according to the limit per a ray

                if bounce_mask.any():
                    # decide how many bounces to calculate
                    brefdirs = refdirs[bounce_mask].reshape(-1, 1, 3)
                    # add noise to simulate roughness
                    outward = L[bounce_mask]
                    noise_rays, mipval = self.brdf_sampler.sample(num_roughness_rays, brefdirs, V[bounce_mask], outward, roughness[bounce_mask]**2, ray_mask)
                    bounce_rays = torch.cat([
                        xyz_sampled[full_bounce_mask[ray_valid]][..., :3].reshape(-1, 1, 3).expand(-1, num_roughness_rays, 3)[ray_mask],
                        noise_rays,
                        # rays_up[full_bounce_mask].reshape(-1, 1, 3).expand(noise_rays.shape)
                    ], dim=-1)
                    n = bounce_rays.shape[0]
                    D = bounce_rays.shape[-1]

                    if recur <= 0 and self.world_bounces > 0:
                        norm_ray_origins = self.rf.normalize_coord(bounce_rays[..., :3])
                        # vis_mask takes in each outgoing ray and predicts whether it will terminate at the background
                        # vis_mask = self.visibility_module.mask(norm_ray_origins.reshape(-1, 3), bounce_rays[:, 3:6].reshape(-1, 3), self.world_bounces, full_bounce_mask, ray_mask, weight)
                        eps = 5e-3
                        vis_mask = ((bounce_rays[..., 0] < eps) & (bounce_rays[..., 1] > -eps)) & (bounce_rays.abs().min(dim=1).values < eps)
                        incoming_light = torch.zeros((n, 3), device=device)
                        # for high sigvis, this implies that the ray terminates and needs to be rendered fully
                        debug[full_bounce_mask] += row_mask_sum(vis_mask.float().reshape(-1, 1), ray_mask).expand(-1, 3)
                        if vis_mask.sum() > 0:
                            # ic(bounce_rays.reshape(-1, D)[vis_mask].shape, mipval.reshape(-1)[vis_mask].shape)
                            incoming_data = self(bounce_rays.reshape(-1, D)[vis_mask], focal, recur=recur+1, white_bg=False,
                                                 start_mipval=mipval.reshape(-1)[vis_mask], override_near=0.05, is_train=is_train,
                                                 ndc_ray=ndc_ray, N_samples=N_samples, tonemap=False)
                            # if not is_train:
                            #     ic(incoming_data['depth_map'].max())
                            incoming_light[vis_mask] = incoming_data['rgb_map']
                            incoming_light[~vis_mask] = self.render_just_bg(bounce_rays.reshape(-1, D)[~vis_mask], mipval.reshape(-1)[~vis_mask])
                        else:
                            incoming_light = self.render_just_bg(bounce_rays.reshape(-1, D), mipval.reshape(-1))
                    else:
                        incoming_light = self.render_just_bg(bounce_rays.reshape(-1, D), mipval.reshape(-1))

                    # self.brdf_sampler.update(bounce_rays[..., :3].reshape(-1, 3), mipval.reshape(-1), incoming_light.reshape(-1, 3))
                    # miplevel = self.bg_module.sa2mip(bounce_rays[..., 3:6], mipval)
                    # debug[full_bounce_mask][..., 0] += miplevel.mean(dim=1) / (self.bg_module.max_mip-1)
                    
                    tinted_ref_rgb = self.brdf(incoming_light,
                            V[bounce_mask], bounce_rays[..., 3:6], outward.reshape(-1, 1, 3),
                            noise_app_features[bounce_mask], roughness[bounce_mask], matprop,
                            bounce_mask, ray_mask)
                    # ic(ray_mask.sum(dim=-1).float().mean(), ray_mask.shape)
                    # s = incoming_light.max(dim=1).values

                    # if not is_train:
                    #     plt.style.use('dark_background')
                    #     plt.scatter(self.brdf_sampler.angs[:100, 0], self.brdf_sampler.angs[:100, 1], c=incoming_light.clip(0, 1).detach().cpu()[0])
                    #     plt.show()

                    # s = incoming_light[:, 0]
                    s = row_mask_sum(incoming_light.detach(), ray_mask) / (ray_mask.sum(dim=1)+1e-8)[..., None]
                    # debug[full_bounce_mask] += s# / (s+1)
                    # ic(tint.mean(dim=0), tinted_ref_rgb.mean(dim=0), incoming_light.mean(dim=0))
                    # debug[full_bounce_mask] += tint[bounce_mask]
                    debug[full_bounce_mask] += 1
                    # reflect_rgb[bounce_mask] = tint[bounce_mask] * tinted_ref_rgb
                    reflect_rgb[bounce_mask] = tint[bounce_mask][..., 0:1] * tinted_ref_rgb
                    # ic(tint.mean(dim=0), tinted_ref_rgb.mean(dim=0), s.mean(dim=0))
                    # reflect_rgb[bounce_mask] = tint[bounce_mask] * s
                    # reflect_rgb[bounce_mask] = tinted_ref_rgb
                    # reflect_rgb[bounce_mask] = s

                    # m = full_bounce_mask.sum(dim=1) > 0
                    # LOGGER.log_rays(rays_chunk[m].reshape(-1, D), recur, dict(depth_map=depth_map.detach()[m]))
                    # LOGGER.log_rays(bounce_rays.reshape(-1, D), recur+1, reflect_data)
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
            # ic(reflect_rgb.mean(), matprop['diffuse'].mean())
            rgb[app_mask] = reflect_rgb + matprop['diffuse']
            # debug[app_mask] = matprop['diffuse']
            tint = tint[..., 0:1]
        else:
            tint = torch.tensor(0.0)
            reflect_rgb = torch.tensor(0.0)
            v_world_normal = world_normal
            diffuse = torch.tensor(0.0)
            roughness = torch.tensor(0.0)
        

        # calculate depth

        # shadow_map = torch.sum(weight * shadows, 1)
        # (N, bundle_size, bundle_size)
        acc_map = torch.sum(weight, 1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        if not is_train:
            with torch.no_grad():
                depth_map = torch.sum(weight * z_vals, 1)
                # depth_map = depth_map + (1. - acc_map) * rays_chunk[whole_valid, -1]
                depth_map = depth_map + (1. - acc_map) * 0

                # view dependent normal map
                # N, 3, 3
                # row_basis = -torch.stack([
                #     -torch.linalg.cross(viewdirs[:, 0], rays_up[:, 0], dim=-1),
                #     viewdirs[:, 0],
                #     rays_up[:, 0],
                # ], dim=1)
                # d_normal_map = torch.matmul(row_basis, d_world_normal_map.unsqueeze(-1)).squeeze(-1)

                v_world_normal_map = row_mask_sum(p_world_normal*pweight[..., None], ray_valid)
                v_world_normal_map = acc_map[..., None] * v_world_normal_map + (1 - acc_map[..., None])
                world_normal_map = row_mask_sum(world_normal*pweight[..., None], ray_valid)
                world_normal_map = acc_map[..., None] * world_normal_map + (1 - acc_map[..., None])

                if weight.shape[1] > 0:
                    inds = ((weight).max(dim=1).indices).clip(min=0)
                    full_xyz_sampled = torch.zeros((B, max_samps, 4), device=device)
                    full_xyz_sampled[ray_valid] = xyz_sampled

                    termination_xyz = full_xyz_sampled[range(full_shape[0]), inds].cpu()
                else:
                    termination_xyz = torch.empty(0, 4)

                # collect statistics about the surface
                # surface width in voxels
                surface_width = app_mask.sum(dim=1)

                # TODO REMOVE
                LOGGER.log_norms_n_rays(xyz_sampled[papp_mask], v_world_normal[papp_mask], weight[app_mask])
                debug_map = (weight[..., None]*debug).sum(dim=1)
            output['depth_map'] = depth_map.detach().cpu()
            output['normal_map'] = v_world_normal_map.detach().cpu()
            output['world_normal_map'] = world_normal_map.detach().cpu()
            output['termination_xyz'] = termination_xyz
            output['debug_map'] = debug_map.detach().cpu()
            output['surf_width'] = surface_width

            if app_mask.any():
                eweight = weight[app_mask][..., None]
                t = tint[..., 0:1].expand(-1, 3)
                tint_map = row_mask_sum(t.detach()*eweight, app_mask).cpu()
                diffuse_map = row_mask_sum(matprop['diffuse'].detach()*eweight, app_mask).cpu()
                roughness_map = row_mask_sum(roughness.reshape(-1, 1).detach()*eweight, app_mask).cpu()
            else:
                tint_map = torch.zeros(rgb_map.shape)
                diffuse_map = torch.zeros(rgb_map.shape)
                roughness_map = torch.zeros((rgb_map.shape[0], 1))
            if app_mask.any() and bounce_mask.any():
                # s = row_mask_sum(incoming_light.detach(), ray_mask) / (ray_mask.sum(dim=1)+1e-8)[..., None]
                s = tinted_ref_rgb
                spec_map = row_mask_sum(s*weight[full_bounce_mask][..., None], full_bounce_mask).cpu()
            else:
                spec_map = torch.zeros(rgb_map.shape)
            output['tint_map'] = tint_map
            output['diffuse_map'] = diffuse_map
            output['spec_map'] = spec_map
            output['roughness_map'] = roughness_map
        else:
            # viewdirs point inward. -viewdirs aligns with p_world_normal. So we want it below 0
            backwards_rays_loss = torch.matmul(viewdirs[ray_valid].reshape(-1, 1, 3), p_world_normal.reshape(-1, 3, 1)).reshape(pweight.shape).clamp(min=0)**2
            backwards_rays_loss = (pweight * backwards_rays_loss).sum() / pweight.sum().clip(min=1e-10)

            midpoint = torch.cat([
                z_vals,
                (2*z_vals[:, -1] - z_vals[:, -2])[:, None],
            ], dim=1)
            # extend the dt artifically to the background
            dt = torch.cat([
                dists,
                0*dists[:, -2:-1]
            ], dim=1)
            full_weight = torch.cat([weight, 1-weight.sum(dim=1, keepdim=True)], dim=1)
            # floater_loss = lossfun_distortion(midpoint, full_weight, dt).clip(min=self.max_floater_loss)
            # TODO REMOVE
            floater_loss = distortion_loss(midpoint, full_weight, dt) if is_train else torch.tensor(0.0, device=device) 
            # floater_loss = torch.tensor(0.0, device=device) 

            align_world_loss = (1-(p_world_normal * world_normal).sum(dim=-1).clamp(max=self.max_normal_similarity))
            # align_world_loss = torch.linalg.norm(p_world_normal - world_normal, dim=-1)
            normal_loss = (pweight * align_world_loss).sum() / B

            # output['diffuse_reg'] = (roughness-0.5).clip(min=0).mean() + tint.clip(min=1e-3).mean()
            output['diffuse_reg'] = tint.clip(min=1e-3).mean()
            # output['diffuse_reg'] = roughness.mean() + reflect_rgb.mean()
            output['normal_loss'] = normal_loss
            output['backwards_rays_loss'] = backwards_rays_loss
            output['floater_loss'] = floater_loss

        # if recur > 0:
        #     ic(weight.sum(), z_vals, xyz_sampled[..., :3], rgb_map.max())

        if tonemap:
            rgb_map = self.tonemap(rgb_map.clip(min=0), noclip=True)
        # ic(weight.mean(), rgb.mean(), rgb_map.mean(), v_world_normal.mean(), sigma.mean(), dists.mean(), alpha.mean())

        if self.bg_module is not None and not white_bg:
            # ic(mipval)
            bg_roughness = torch.zeros(B, 1, device=device) if start_mipval is None else start_mipval
            bg = self.bg_module(viewdirs[:, 0, :], bg_roughness).reshape(-1, 3)
            if tonemap:
                bg = self.tonemap(bg, noclip=True)
            if self.detach_bg and recur > 0:
                bg = bg.detach()
            rgb_map = rgb_map + \
                (1 - acc_map[..., None]) * bg
        else:
            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1 - acc_map[..., None])

        return dict(
            **output,
            rgb_map=rgb_map,
            # weight_slice=weight_slice,
            recur=recur,
            acc_map=acc_map.detach().cpu(),
            roughness=roughness.mean(),

            color_count=app_mask.detach().sum(),
            bounce_count=bounce_count,
            whole_valid=whole_valid, 
        )
