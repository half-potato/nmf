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
from modules import calc_distortion_loss, row_mask_sum
from mutils import normalize
from models import sh

LOGGER = Logger(enable=False)
FIXED_SPHERE = False
FIXED_RETRO = False

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    # alpha = 1. - torch.exp((-sigma*dist).clip(min=-1e10, max=0))
    alpha = 1. - torch.exp(-sigma*dist)

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
                 sampler, diffuse_module=None, brdf_sampler=None, brdf=None, tonemap=None, normal_module=None, ref_module=None, bg_module=None,
                 visibility_module=None, grid_size=None, bright_sampler=None,
                 alphaMask=None, transmittance_thres=1, 
                 infinity_border=False, max_brdf_rays=[524288],
                 rayMarch_weight_thres=0.0001, recur_weight_thres=1e-3,detach_inter=False, percent_bright=0.1, bg_noise=0, bg_noise_decay=0.999, use_predicted_normals=True,
                 anoise=0.0, anoise_decay=1, normalize_brdf=True, orient_world_normals=False,
                 selector=None, cold_start_bg_iters = 0, eval_batch_size=512, rough_light=False,
                 lr_scale=1, diffuse_dropout=0, detach_N_iters=0,
                 **kwargs):
        super(TensorNeRF, self).__init__()
        self.rf = rf(aabb=aabb, grid_size=grid_size)
        self.ref_module = ref_module(in_channels=self.rf.app_dim) if ref_module is not None else None
        self.normal_module = normal_module(in_channels=self.rf.app_dim) if normal_module is not None else None
        bound = aabb.abs().max()
        self.diffuse_module = diffuse_module(in_channels=self.rf.app_dim) if diffuse_module is not None else None
        al = self.diffuse_module.allocation if self.diffuse_module is not None else 0
        self.normal_module = normal_module(in_channels=self.rf.app_dim-al) if normal_module is not None else None
        al += self.normal_module.allocation if self.normal_module is not None else 0
        self.ref_module = ref_module(in_channels=self.rf.app_dim-al) if ref_module is not None else None
        self.brdf = brdf(in_channels=self.rf.app_dim-al) if brdf is not None else None
        self.brdf_sampler = brdf_sampler if brdf_sampler is None else brdf_sampler(max_samples=1024)
        self.visibility_module = visibility_module(in_channels=self.rf.app_dim-al, bound=bound) if (visibility_module is not None) and len(max_brdf_rays) > 1 else None
        self.sampler = sampler(near_far=near_far, aabb=aabb)
        self.bright_sampler = bright_sampler if bright_sampler is None else bright_sampler(max_samples=int(100*percent_bright+1), cold_start_bg_iters=cold_start_bg_iters)
        self.selector = selector(percent_bright=percent_bright) if selector is not None else None
        self.bg_module = bg_module
        if tonemap is None:
            self.tonemap = SRGBTonemap()
        else:
            self.tonemap = tonemap

        self.max_brdf_rays = max_brdf_rays
        self.lr_scale = lr_scale
        self.diffuse_dropout = diffuse_dropout
        self.normalize_brdf = normalize_brdf
        self.bg_noise = bg_noise
        self.bg_noise_decay = bg_noise_decay
        self.alphaMask = alphaMask
        self.infinity_border = infinity_border
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.anoise = anoise
        self.allow_transmit = False
        self.transmittance_thres = transmittance_thres
        self.recur_weight_thres = recur_weight_thres
        self.eval_batch_size = eval_batch_size
        self.rough_light = rough_light

        self.cold_start_bg_iters = cold_start_bg_iters
        self.detach_bg = True
        self.detach_N_iters = detach_N_iters
        self.detach_N = True
        self.detach_inter = detach_inter
        self.anoise_decay = anoise_decay

        self.use_predicted_normals = use_predicted_normals
        self.orient_world_normals = orient_world_normals | (not use_predicted_normals)
        # self.sampler.update(self.rf, init=True)
        # self.sampler2.update(self.rf, init=True)


    def get_device(self):
        return self.rf.units.device

    def get_optparam_groups(self, lr_bg=0.025, lr_scale=1):
        grad_vars = []
        grad_vars += self.rf.get_optparam_groups(self.lr_scale)
        if isinstance(self.normal_module, torch.nn.Module):
            grad_vars += [{'params': self.normal_module.parameters(),
                           'lr': self.normal_module.lr*self.lr_scale}]
        if self.ref_module is not None:
            grad_vars += [{'params': list(self.ref_module.parameters()), 'lr': lr_scale*self.ref_module.lr, 'beta': [0.9, 0.999]}]
        if isinstance(self.diffuse_module, torch.nn.Module):
            grad_vars += [{'params': self.diffuse_module.parameters(),
                           'lr': self.diffuse_module.lr*self.lr_scale}]
        if isinstance(self.brdf, torch.nn.Module):
            grad_vars += [{'params': self.brdf.parameters(),
                           'lr': self.brdf.lr*self.lr_scale}]
        if isinstance(self.bg_module, torch.nn.Module):
            grad_vars += self.bg_module.get_optparam_groups(self.lr_scale)
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
        del ckpt['state_dict']['brdf_sampler.angs']
        near_far = near_far if near_far is not None else [1, 6]
        if 'rf.grid_size' in ckpt['state_dict']:
            grid_size = list(ckpt['state_dict']['rf.grid_size'])
        else:
            grid_size = None
        rf = hydra.utils.instantiate(config)(aabb=aabb, near_far=near_far, grid_size=grid_size)
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

    def check_schedule(self, iter, batch_mul):
        require_reassignment = False
        require_reassignment |= self.sampler.check_schedule(iter, batch_mul, self.rf)
        require_reassignment |= self.rf.check_schedule(iter, batch_mul)
        if self.bright_sampler is not None:
            self.bright_sampler.check_schedule(iter, batch_mul, self.bg_module)
        if require_reassignment:
            self.sampler.update(self.rf, init=True)
        if iter > batch_mul*self.cold_start_bg_iters:
            self.detach_bg = False
        if iter > batch_mul*self.detach_N_iters:
            self.detach_N = False
        self.anoise *= self.anoise_decay
        self.bg_noise *= self.bg_noise_decay
        return require_reassignment

    def render_env_sparse(self, ray_origins, env_dirs, roughness: float):
        B, M = env_dirs.shape[:2]
        ray_origins = torch.cat([ray_origins, roughness*torch.ones((B, 1), device=self.get_device())], dim=-1)
        norm_ray_origins = self.rf.normalize_coord(ray_origins)
        app_features = self.rf.compute_appfeature(ray_origins)
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

        if self.ref_module is not None:
            roughness = 1/np.pi*torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            # roughness = matprop['roughness'] if roughness is None else roughness * torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            viewdotnorm = torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            envmap = (self.ref_module(xyz_samp, staticdir, app_features, refdirs=ang_vecs.reshape(
                -1, 3), roughness=roughness, viewdotnorm=viewdotnorm)).reshape(res, 2*res, 3)
        else:
            envmap = torch.zeros(res, 2*res, 3)
        if self.diffuse_module is not None:
            color, tint, matprop = self.diffuse_module(xyz_samp, ang_vecs.reshape(-1, 3), app_features)
            color = (color).reshape(res, 2*res, 3)/2
        else:
            color = torch.zeros(res, 2*res, 3)
        
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
            validsigma = self.rf.compute_densityfeature(xyz_g, activate=False)

            # compute normal
            grad_outputs = torch.ones_like(validsigma)
            g = grad(validsigma, xyz_g, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)
            # n = torch.linalg.norm(g[0][:, :3], dim=-1)
            # ic(g[0][:, :3].abs().max())
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
                            psigma = self.rf.compute_densityfeature(xyz_sampled)
                        else:
                            psigma, all_app_features = self.rf.compute_feature(xyz_sampled)
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
        # app_features = self.rf.compute_appfeature(origins)
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
                is_train=False, ndc_ray=False, N_samples=-1, tonemap=True, draw_debug=True):
        # rays_chunk: (N, (origin, viewdir, ray_up))
        output = {}
        eps = torch.finfo(torch.float32).eps

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

        all_app_features = None
        pred_norms = torch.zeros((M, 3), device=device)
        if FIXED_SPHERE:
            sigma[ray_valid] = torch.where(torch.linalg.norm(xyz_sampled, dim=-1) < 0.44, 99999999.0, 0.0)
        elif FIXED_RETRO:
            mask1 = torch.linalg.norm(xyz_sampled, dim=-1, ord=torch.inf) < 0.613
            mask2 = (xyz_sampled[..., 0] < 0) & (xyz_sampled[..., 1] > 0)
            sigma[ray_valid] = torch.where(mask1 & ~mask2, 99999999.0, 0.0)
        else:
            if ray_valid.any():
                if self.rf.separate_appgrid:
                    psigma = self.rf.compute_densityfeature(xyz_sampled)
                else:
                    psigma, all_app_features = self.rf.compute_feature(xyz_sampled)
                sigma[ray_valid] = psigma


        if self.rf.contract_space and self.infinity_border:
            at_infinity = self.at_infinity(xyz_normed)
            sigma[at_infinity] = 100

        # weight: [N_rays, N_samples]
        # ic((dists * self.rf.distance_scale).mean())
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.rf.distance_scale)

        # app stands for appearance
        pweight = weight[ray_valid]
        thres = self.rayMarch_weight_thres if recur == 0 else self.recur_weight_thres
        app_mask = (weight > thres)
        papp_mask = app_mask[ray_valid]

        # if self.visibility_module is not None:
        #     self.visibility_module.ray_update(xyz_normed, viewdirs[ray_valid], app_mask, ray_valid)

        debug = torch.zeros((B, n_samples, 3), dtype=torch.float, device=device, requires_grad=False)
        brdf_rgb = torch.zeros((B, n_samples, 3), dtype=torch.float, device=device, requires_grad=False)
        bounce_count = 0

        rgb = torch.zeros((*full_shape[:2], 3), device=device, dtype=weight.dtype)
        bounce_mask = torch.zeros((1), dtype=bool)
        brdf_brightness = torch.tensor(0.0)

        if app_mask.any():
            #  Compute normals for app mask
            
            app_xyz = xyz_sampled[papp_mask]

            # TODO REMOVE
            norms = self.calculate_normals(app_xyz)
            world_normal[papp_mask] = norms

            app_norm_xyz = xyz_normed[papp_mask]

            if all_app_features is None:
                app_features = self.rf.compute_appfeature(app_xyz)
            else:
                app_features = all_app_features[papp_mask]
                # _, app_features = self.rf.compute_feature(app_norm_xyz)

            noise_app_features = (app_features + torch.randn_like(app_features) * self.anoise)

            # get base color of the point
            diffuse, tint, matprop = self.diffuse_module(
                app_norm_xyz, viewdirs[app_mask], noise_app_features)
            # f0 = matprop['f0']
            r1 = matprop['r1']
            r2 = matprop['r2']
            # r1 = 1 - weight[app_mask][..., None].clip(min=eps).sqrt() * (1-r1)
            # r2 = 1 - weight[app_mask][..., None].clip(min=eps).sqrt() * (1-r2)

            # ic(diffuse.mean(dim=0))
            if self.diffuse_module is not None:
                app_features = app_features[..., self.diffuse_module.allocation:]
                noise_app_features = noise_app_features[..., self.diffuse_module.allocation:]
            # diffuse = diffuse * bg_color
            # diffuse = diffuse.type(rgb.dtype)

            # interpolate between the predicted and world normals
            if self.normal_module is not None:
                pred_norms = torch.zeros_like(pred_norms)
                pred_norms[papp_mask] = self.normal_module(app_norm_xyz, app_features, world_normal[papp_mask])
                v_world_normal = pred_norms if self.use_predicted_normals else world_normal
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

                app_features = app_features[..., self.normal_module.allocation:]
                noise_app_features = noise_app_features[..., self.normal_module.allocation:]
            else:
                v_world_normal = world_normal


            # calculate reflected ray direction
            V = -viewdirs[app_mask]
            N = v_world_normal[papp_mask]
            # VdotN = (V * N).sum(-1, keepdim=True)
            # app_weight = weight[app_mask][..., None]
            # N = normalize(N + (1-app_weight)**2 * (torch.rand_like(N)*2 - 1))
            # N = torch.where(torch.rand_like(app_weight)/4 < app_weight, N, V)
            VdotN = (V * N).sum(-1, keepdim=True)
            refdirs = 2 * VdotN * N - V

            if self.bg_module is not None:
                # compute spherical harmonic coefficients for the background
                # if self.detach_bg:
                coeffs, conv_coeffs = self.bg_module.get_spherical_harmonics(100)
                evaled = sh.eval_sh_bases(coeffs.shape[0], N)
                E = (conv_coeffs.reshape(1, -1, 3) * evaled.reshape(evaled.shape[0], -1, 1)).sum(dim=1).detach()
                # diffuse = diffuse * E.detach()# * np.pi

            reflect_rgb = torch.zeros_like(diffuse)
            # roughness = torch.min(r1, r2).squeeze(-1)
            roughness = r1.squeeze(-1)
            # roughness = 1e-3*torch.ones_like(roughness)
            # roughness = torch.where((xyz_sampled[..., 0].abs() < 0.15) | (xyz_sampled[..., 1].abs() < 0.15), 0.30, 0.15)[papp_mask]
            if self.ref_module is not None:
                viewdotnorm = (viewdirs[app_mask]*N).sum(dim=-1, keepdim=True)
                ref_col = self.ref_module(
                    app_norm_xyz, viewdirs[app_mask],
                    noise_app_features, refdirs=refdirs,
                    roughness=roughness, viewdotnorm=viewdotnorm)
                reflect_rgb = tint * ref_col
                debug[app_mask] += ref_col / (ref_col + 1)
                rgb[app_mask] = (reflect_rgb + diffuse).clip(0, 1)
            elif self.brdf is not None:
                num_brdf_rays = self.max_brdf_rays[recur] // weight.shape[0]
                bounce_mask, full_bounce_mask, inv_full_bounce_mask, ray_mask, bright_mask = self.selector(
                        app_mask, weight.detach(), VdotN, 1-roughness.detach(), num_brdf_rays)
                tm_mask = torch.zeros_like(bounce_mask)

                ray_xyz = app_xyz[bounce_mask][..., :3].reshape(-1, 1, 3).expand(-1, ray_mask.shape[1], 3)
                if bounce_mask.any() and ray_mask.any() and ray_xyz.shape[0] == ray_mask.shape[0]:
                    bN = N[bounce_mask]
                    if self.detach_inter:
                        bN.detach_()
                    bV = V[bounce_mask]
                    r1 = r1[bounce_mask]
                    r2 = r2[bounce_mask]
                    L, row_world_basis = self.brdf_sampler.sample(
                            bV, bN,
                            r1**2, r2**2, ray_mask)

                    # Sample bright spots
                    if self.bright_sampler is not None and self.bright_sampler.is_initialized():
                        wN = world_normal[papp_mask][bounce_mask]
                        bL, bright_mask = self.bright_sampler.sample(bV, wN, ray_mask, bright_mask)
                        pbright_mask = bright_mask[ray_mask]
                        L[pbright_mask] = bL[bright_mask]

                    eV = bV.reshape(-1, 1, 3).expand(-1, ray_mask.shape[1], 3)[ray_mask]
                    eN = bN.reshape(-1, 1, 3).expand(-1, ray_mask.shape[1], 3)[ray_mask]
                    ea = roughness.reshape(-1, 1)[bounce_mask].expand(ray_mask.shape)[ray_mask]

                    H = normalize((eV+L)/2)
                    mipval = self.brdf_sampler.calculate_mipval(H, eV, eN, ray_mask, ea**2)
                    diffvec = torch.matmul(row_world_basis.permute(0, 2, 1), L.unsqueeze(-1)).squeeze(-1)
                    halfvec = torch.matmul(row_world_basis.permute(0, 2, 1), H.unsqueeze(-1)).squeeze(-1)

                    bounce_rays = torch.cat([
                        ray_xyz[ray_mask],
                        L,
                    ], dim=-1)
                    n = bounce_rays.shape[0]
                    D = bounce_rays.shape[-1]
                    if recur < len(self.max_brdf_rays)-1:
                        # norm_ray_origins = self.rf.normalize_coord(bounce_rays[..., :3])
                        # # vis_mask takes in each outgoing ray and predicts whether it will terminate at the background
                        # if self.visibility_module is not None and self.visibility_module.is_initialized():
                        #     vis_mask = self.visibility_module.mask(norm_ray_origins.reshape(-1, 3), bounce_rays[:, 3:6].reshape(-1, 3), self.world_bounces, full_bounce_mask, ray_mask, weight)
                        # else:
                        #     vis_mask = torch.ones_like(bounce_rays[..., 0], dtype=bool)
                        # # eps = 5e-3
                        # # vis_mask = ((bounce_rays[..., 0] < eps) & (bounce_rays[..., 1] > -eps)) & (bounce_rays.abs().min(dim=1).values < eps)
                        # incoming_light = torch.zeros((n, 3), device=device)
                        # # for high sigvis, this implies that the ray terminates and needs to be rendered fully
                        # # debug[full_bounce_mask] += row_mask_sum(vis_mask.float().reshape(-1, 1), ray_mask).expand(-1, 3)
                        #
                        # if vis_mask.sum() > 0:
                        #     incoming_data = self(bounce_rays.reshape(-1, D)[vis_mask], focal, recur=recur+1, white_bg=False,
                        #                          start_mipval=mipval.reshape(-1)[vis_mask], override_near=self.rf.stepSize*5, is_train=is_train,
                        #                          ndc_ray=False, N_samples=N_samples, tonemap=False)
                        #     incoming_light[vis_mask] = incoming_data['rgb_map']
                        #     incoming_light[~vis_mask] = self.render_just_bg(bounce_rays.reshape(-1, D)[~vis_mask], mipval.reshape(-1)[~vis_mask])
                        # else:
                        #     incoming_light = self.render_just_bg(bounce_rays.reshape(-1, D), mipval.reshape(-1))
                        incoming_data = self(bounce_rays.reshape(-1, D), focal, recur=recur+1, white_bg=False,
                                             start_mipval=mipval.reshape(-1), override_near=self.rf.stepSize*5, is_train=is_train,
                                             ndc_ray=False, N_samples=N_samples, tonemap=False, draw_debug=False)
                        incoming_light = incoming_data['rgb_map']
                    else:
                        incoming_light = self.render_just_bg(bounce_rays.reshape(-1, D), mipval.reshape(-1))

                    # ic(incoming_light.mean())
                    n, m = ray_mask.shape
                    efeatures = noise_app_features[bounce_mask].reshape(n, 1, -1).expand(n, m, -1)[ray_mask]
                    eroughness = roughness[bounce_mask].reshape(-1, 1).expand(n, m)[ray_mask].reshape(-1, 1)
                    brdf_weight = self.brdf(eV, L, eN, halfvec, diffvec, efeatures, eroughness)
                    # brdf_weight = self.brdf(eV, L.detach(), eN.detach(), halfvec.detach(), diffvec.detach(), efeatures, eroughness.detach()) #*(1-1e-2) + 1e-2
                    if self.normalize_brdf:
                        norm = row_mask_sum(brdf_weight, ray_mask).clip(min=1.0) #.mean(dim=-1, keepdim=True)
                    else:
                        norm = (ray_mask.sum(dim=1)+1e-8)[..., None]
                    _brdf_rgb = row_mask_sum(brdf_weight, ray_mask) / norm
                    brdf_rgb[full_bounce_mask] = _brdf_rgb
                    brdf_brightness = _brdf_rgb.mean()

                    # spec_color = row_mask_sum(incoming_light * brdf_weight, ray_mask) / norm
                    # tinted_ref_rgb = spec_color
                    # if self.detach_bg:
                    #     tinted_ref_rgb.detach_()

                    # TODO REMOVE
                    spec_color = row_mask_sum(incoming_light, ray_mask) / norm
                    tinted_ref_rgb = tint[bounce_mask] * spec_color
                    if self.detach_bg:
                        tinted_ref_rgb.detach_()
                    reflect_rgb[bounce_mask] = tinted_ref_rgb

                    # ic(_brdf_rgb.mean(dim=0), tinted_ref_rgb.mean(dim=0), incoming_light.mean(dim=0), diffuse.mean(dim=0))

                    # s = row_mask_sum(incoming_light.detach(), ray_mask) / (ray_mask.sum(dim=1)+1e-8)[..., None]
                    # debug[full_bounce_mask] += s# / (s+1)
                    # debug[full_bounce_mask] += ray_mask.sum(dim=1, keepdim=True)
                    # debug[full_bounce_mask] += bright_mask.sum(dim=1, keepdim=True)
                    # reflect_rgb[~bounce_mask] = E[~bounce_mask].detach()

                if (~bounce_mask).any() and self.rough_light:
                    nb_efeatures = app_features[~bounce_mask]
                    nb_eroughness = roughness[~bounce_mask]
                    nb_L = normalize(N[~bounce_mask] + normalize(2*torch.rand((nb_efeatures.shape[0], 3), device=device)-1))
                    # nb_halfvec = normalize(V[~bounce_mask] + N[~bounce_mask])
                    nb_halfvec = normalize(V[~bounce_mask] + nb_L)
                    nb_diffvec = torch.zeros((nb_efeatures.shape[0], 3), device=device)
                    nb_diffvec[:, 2] = 1
                    # nb_diffvec = torch.matmul(row_world_basis.permute(0, 2, 1), L.unsqueeze(-1)).squeeze(-1)
                    nb_brdf_weight = self.brdf(V[~bounce_mask], nb_L.detach(), N[~bounce_mask].detach(), nb_halfvec.detach(), nb_diffvec.detach(), nb_efeatures, nb_eroughness)
                    # nb_ref_col = tint[~bounce_mask].detach() * nb_brdf_weight.detach() * E[~bounce_mask].detach()
                    nb_ref_col = nb_brdf_weight.detach() * E[~bounce_mask].detach()
                    reflect_rgb[~bounce_mask] = nb_ref_col
                    inv_full_bounce_mask = torch.zeros_like(app_mask)
                    ainds, ajinds = torch.where(app_mask)
                    inv_full_bounce_mask[ainds[~bounce_mask], ajinds[~bounce_mask]] = 1
                    brdf_rgb[inv_full_bounce_mask] = nb_brdf_weight

                # bad_mask = VdotN < 0
                # vdotn = VdotN[bad_mask].reshape(-1, 1)
                # reflect_rgb[bad_mask.squeeze(-1)] = tint[bad_mask.squeeze(-1)].detach()*((-vdotn).clip(min=0)**2*torch.rand_like(vdotn))
                # reflect_rgb[bad_mask.squeeze(-1)] = ((-vdotn).clip(min=0)**2*torch.rand((vdotn.shape[0], 1), device=device))
                # reflect_rgb[bad_mask.squeeze(-1)] = ((-vdotn).clip(min=0)**2*torch.randn((vdotn.shape[0], 3), device=device))
                # debug[full_bounce_mask] += 1
                debug[app_mask] = (-VdotN).clip(min=0)**2
                if self.diffuse_dropout > 0 and is_train:
                    dropout = torch.rand((diffuse.shape[0], 1), device=device) < self.diffuse_dropout
                    rgb[app_mask] = reflect_rgb + dropout * diffuse
                else:
                    rgb[app_mask] = reflect_rgb + diffuse
                # ic(rgb[app_mask].mean(dim=0), reflect_rgb.mean(dim=0), reflect_rgb[bounce_mask].mean(dim=0), diffuse.mean(dim=0), incoming_light.mean(dim=0))

                bounce_count = bounce_mask.sum()
            else:
                rgb[app_mask] = diffuse
            # this is a modified rendering equation where the emissive light and light under the integral is all multiplied by the base color
            # in addition, the light is interpolated between emissive and reflective
            # ic(reflect_rgb.mean(), diffuse.mean())
            # debug[app_mask] = diffuse
            tint = tint
        else:
            tint = torch.tensor(0.0)
            reflect_rgb = torch.tensor(0.0)
            v_world_normal = world_normal
            diffuse = torch.tensor(0.0)
            roughness = torch.tensor(0.0)
        

        # calculate depth

        # shadow_map = torch.sum(weight * shadows, 1)
        # (N, bundle_size, bundle_size)
        if recur > 0 and self.detach_inter:
            weight = weight.detach()
            bg_weight = bg_weight.detach()

        acc_map = torch.sum(weight, 1)
        rgb_map = torch.sum(weight[..., None] * rgb.clip(min=0, max=1), -2)
        if not is_train and draw_debug:
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

                world_normal_map = row_mask_sum(world_normal*pweight[..., None], ray_valid)
                world_normal_map = acc_map[..., None] * world_normal_map + (1 - acc_map[..., None])
                pred_norm_map = row_mask_sum(pred_norms*pweight[..., None], ray_valid)
                # ic(pred_norms, pred_norms.mean(dim=0), pred_norm_map.mean(dim=0), pweight[..., None].mean(dim=0))
                pred_norm_map = acc_map[..., None] * pred_norm_map + (1 - acc_map[..., None])

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
                # surface_width = (weight.max(dim=1).values*255).int()
                # filtw = weight[acc_map > 0.9].detach().cpu()
                # if filtw.shape[0] > 10:
                #     hist = filtw[random.randint(0, filtw.shape[0])]
                #     t1 = torch.max(torch.where(hist > 1e-4)[0])
                #     t0 = torch.min(torch.where(hist > 1e-4)[0])
                #     px.bar(x=torch.arange(hist.shape[0])[t0:t1], y=hist[t0:t1]).show()
                #     assert(False)

                # TODO REMOVE
                LOGGER.log_norms_n_rays(xyz_sampled[papp_mask], v_world_normal[papp_mask], weight[app_mask])
                # o_world_normal = world_normal if self.orient_world_normals else pred_norms
                # ori_loss = torch.matmul(viewdirs[ray_valid].reshape(-1, 1, 3).detach(), o_world_normal.reshape(-1, 3, 1)).reshape(pweight.shape).clamp(min=0)**2
                # debug[ray_valid] = ori_loss.reshape(-1, 1).expand(-1, 3)
                debug_map = (weight[..., None]*debug).sum(dim=1)
                # calculate cross section
                pcs_mask = xyz_normed[..., 2] < 0
                full_cs_mask = torch.zeros_like(weight, dtype=bool)
                full_cs_mask[ray_valid] = pcs_mask
                cross_section = torch.sum(full_cs_mask[..., None] * weight[..., None] * rgb.clip(min=0, max=1), -2)
            output['cross_section'] = cross_section.detach().cpu()
            output['depth_map'] = depth_map.detach().cpu()
            output['world_normal_map'] = world_normal_map.detach().cpu()
            output['normal_map'] = pred_norm_map.detach().cpu()
            output['termination_xyz'] = termination_xyz
            output['debug_map'] = debug_map.detach().cpu()
            output['surf_width'] = surface_width


            if app_mask.any():
                eweight = weight[app_mask][..., None]
                # t = tint[..., 0:1].expand(-1, 3)
                t = tint
                tint_map = row_mask_sum(t.detach()*eweight, app_mask).cpu()
                diffuse_map = row_mask_sum(diffuse.detach()*eweight, app_mask).cpu()
                roughness_map = row_mask_sum(roughness.reshape(-1, 1).detach()*eweight, app_mask).cpu()
            else:
                tint_map = torch.zeros(rgb_map.shape)
                diffuse_map = torch.zeros(rgb_map.shape)
                roughness_map = torch.zeros((rgb_map.shape[0], 1))
            spec_map = torch.zeros(rgb_map.shape)
            r0_map = torch.zeros(rgb_map.shape)
            diffuse_light_map = torch.zeros(rgb_map.shape)
            transmit_map = torch.zeros(rgb_map.shape)
            brdf_map = (weight[..., None]*brdf_rgb).sum(dim=1)
            if app_mask.any():
                spec_map = row_mask_sum(reflect_rgb*weight[app_mask][..., None], app_mask).cpu()
            if app_mask.any() and self.bg_module is not None:
                diffuse_light_map = row_mask_sum(E*weight[app_mask][..., None], app_mask).cpu()
            if app_mask.any() and self.brdf is not None and bounce_mask.any() and ray_mask.any():
                s = row_mask_sum(incoming_light, ray_mask) / (ray_mask.sum(dim=1)+1e-16)[..., None]
                # s = tinted_ref_rgb
                # s = brdf_rgb
                # r0_map = row_mask_sum(meanR0.expand(-1, 3)*weight[full_bounce_mask][..., None], full_bounce_mask).cpu()
                # s2 = row_mask_sum(incoming_light[bright_mask[ray_mask]], bright_mask) / (bright_mask.sum(dim=1)+1e-16)[..., None]
                # diffuse_light_map = row_mask_sum(s2*weight[full_bounce_mask][..., None], full_bounce_mask).cpu()

                spec_map = row_mask_sum(s*weight[full_bounce_mask][..., None], full_bounce_mask).cpu()
                if tm_mask.any():
                    full_tm_mask = torch.zeros_like(full_bounce_mask)
                    full_tm_mask[app_mask] = tm_mask
                    transmit_map = row_mask_sum(tm_light*weight[app_mask][tm_mask][..., None], full_tm_mask).cpu()
            output['tint_map'] = tint_map
            output['diffuse_map'] = diffuse_map
            output['spec_map'] = spec_map
            output['r0_map'] = r0_map
            output['transmitted'] = transmit_map
            output['diffuse_light_map'] = diffuse_light_map
            output['brdf_map'] = brdf_map
            output['roughness_map'] = roughness_map
        elif recur == 0:
            # viewdirs point inward. -viewdirs aligns with pred_norms. So we want it below 0
            o_world_normal = world_normal if self.orient_world_normals else pred_norms
            aweight = pweight[papp_mask]
            NdotV = (-viewdirs[app_mask].reshape(-1, 3).detach() * o_world_normal[papp_mask].reshape(-1, 3)).sum(dim=-1)
            ori_loss = (aweight * NdotV.clamp(max=0)**2).sum() / B

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
            # TODO REMOVE
            distortion_loss = calc_distortion_loss(midpoint, full_weight, dt)
            # distortion_loss = torch.tensor(0.0, device=device) 

            if self.use_predicted_normals:
                align_world_loss = 2*(1-(pred_norms[papp_mask] * world_normal[papp_mask]).sum(dim=-1))#**0.5
                # ic(pred_norms.mean(), world_normal.mean(), align_world_loss.mean(), align_world_loss.shape)
                prediction_loss = (aweight * align_world_loss).sum() / B
            else:
                prediction_loss = torch.tensor(0.0)

            # output['diffuse_reg'] = (roughness-0.5).clip(min=0).mean() + tint.clip(min=1e-3).mean()
            # output['diffuse_reg'] = tint.clip(min=1e-3).mean()
            if self.bg_module is not None:
                envmap_brightness = self.bg_module.mean_color().mean()
                if self.detach_bg:
                    envmap_brightness.detach_()
                output['envmap_reg'] = (envmap_brightness).clip(min=0)
            else:
                output['envmap_reg'] = torch.tensor(0.0)

            output['brdf_reg'] = -brdf_brightness
            output['diffuse_reg'] = -roughness.mean()
            output['prediction_loss'] = prediction_loss
            output['ori_loss'] = ori_loss
            output['distortion_loss'] = distortion_loss

        # if recur > 0:
        #     ic(weight.sum(), z_vals, xyz_sampled[..., :3], rgb_map.max())

        if tonemap:
            rgb_map = self.tonemap(rgb_map.clip(min=0), noclip=True)
        # ic(weight.mean(), rgb.mean(), rgb_map.mean(), v_world_normal.mean(), sigma.mean(), dists.mean(), alpha.mean())

        if self.bg_module is not None and not white_bg:
            bg_roughness = -100*torch.ones(B, 1, device=device) if start_mipval is None else start_mipval
            bg = self.bg_module(viewdirs[:, 0, :], bg_roughness).reshape(-1, 3)
            if tonemap:
                bg = self.tonemap(bg, noclip=True)
            if self.detach_bg and recur > 0:
                bg = bg.detach()
            rgb_map = rgb_map + \
                (1 - acc_map[..., None]) * bg
        else:
            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                if output_alpha is not None and self.bg_noise > 0:
                    noise = (torch.rand((1, 1), device=device) > 0.5).float()*output_alpha[:, None] + (1-output_alpha[:, None])
                    # noise = (torch.rand((*acc_map.shape, 1), device=device) > 0.5).float()*output_alpha[:, None] + (1-output_alpha[:, None])
                else:
                    noise = 1-torch.rand((*acc_map.shape, 3), device=device)*self.bg_noise
                # noise = 1-torch.rand((*acc_map.shape, 3), device=device)*self.bg_noise
                rgb_map = rgb_map + (1 - acc_map[..., None]) * noise
            # if white_bg:
            #     if True:
            #         bg_col = torch.rand((1, 3), device=device).clip(min=torch.finfo(torch.float32).eps).sqrt()
            #         # rgb_map = rgb_map + (1 - acc_map[..., None]) * torch.rand_like(rgb_map)
            #         rgb_map = rgb_map + (1 - acc_map[..., None]) * torch.where(torch.rand_like(acc_map[..., None]) < 0.5, 0, 1)
            #         # rgb_map = rgb_map + (1 - acc_map[..., None]) * bg_col
            #     else:
            #         rgb_map = rgb_map + (1 - acc_map[..., None])

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
