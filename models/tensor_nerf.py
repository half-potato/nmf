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
from samplers.alphagrid import AlphaGridSampler
from modules import distortion_loss, row_mask_sum

LOGGER = Logger(enable=False)
FIXED_SPHERE = True


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

def lossfun_distortion(midpoint, full_weight, dt):
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    # extend the mipoint artifically to the background
    dut = torch.abs(midpoint[..., :, None] - midpoint[..., None, :])
    # mp = midpoint[..., None]
    # dut = torch.cdist(mp, mp, p=1)
    # loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)
    B = dt.shape[0]
    loss_inter = torch.einsum('bj,bk,bjk', full_weight.reshape(B, -1), full_weight.reshape(B, -1), dut)
    # ic(dt.shape, full_weight.shape)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(full_weight**2 * dt) / 3
    # ic(1, loss_inter, loss_intra)

    return loss_inter + loss_intra

def lossfun_distortion2(t, w, dt):
    device = w.device
    B, n_samples = w.shape
    full_weight = torch.cat([w, 1-w.sum(dim=1, keepdim=True)], dim=1)
    #
    # midpoint = t
    # fweight = torch.abs(midpoint[..., :, None] - midpoint[..., None, :])
    # # # ut = (z_vals[:, 1:] + z_vals[:, :-1])/2
    # #
    # loss_inter = torch.einsum('bj,bk,jk', full_weight.reshape(B, -1), full_weight.reshape(B, -1), fweight)
    # loss_intra = (w**2 * dt).sum(dim=1).sum()/3
    #
    # # this one consumes too much memory

    S = torch.linspace(0, 1, n_samples+1, device=device).reshape(-1, 1)
    # S = t[0, :].reshape(-1, 1)
    fweight = (S - S.T).abs()
    # ut = (z_vals[:, 1:] + z_vals[:, :-1])/2

    floater_loss_1 = torch.einsum('bj,bk,jk', full_weight.reshape(B, -1), full_weight.reshape(B, -1), fweight)
    floater_loss_2 = (full_weight**2).sum()/3/n_samples
    # ic(fweight)

    # ic(floater_loss_1, floater_loss_2)
    return floater_loss_1 + floater_loss_2

class TensorNeRF(torch.nn.Module):
    def __init__(self, rf, aabb, diffuse_module, sampler, brdf_sampler=None, brdf=None, tonemap=None, normal_module=None, ref_module=None, bg_module=None,
                 alphaMask=None, specularity_threshold=0.005, max_recurs=0,
                 max_normal_similarity=1, infinity_border=False, min_refraction=1.1, enable_refraction=True,
                 alphaMask_thres=0.001, rayMarch_weight_thres=0.0001,
                 max_bounce_rays=4000, roughness_rays=3, bounce_min_weight=0.001, appdim_noise_std=0.0,
                 world_bounces=0, selector=None,
                 update_sampler_list=[5000], max_floater_loss=6, **kwargs):
        super(TensorNeRF, self).__init__()
        self.rf = rf(aabb=aabb)
        self.ref_module = ref_module(in_channels=self.rf.app_dim) if ref_module is not None else None
        self.normal_module = normal_module(in_channels=self.rf.app_dim) if normal_module is not None else None
        self.diffuse_module = diffuse_module(in_channels=self.rf.app_dim)
        self.brdf = brdf(in_channels=self.rf.app_dim) if brdf is not None else None
        self.brdf_sampler = brdf_sampler if brdf_sampler is None else brdf_sampler(num_samples=roughness_rays)
        self.sampler = sampler
        self.sampler2 = AlphaGridSampler(near_far=[2, 6])
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
    def load(ckpt, config=None, **kwargs):
        config = ckpt['config'] if config is None else config
        aabb = ckpt['state_dict']['rf.aabb']
        rf = hydra.utils.instantiate(config)(aabb=aabb)
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
            self.sampler.update(self.rf)
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
                recur=0, override_near=None, output_alpha=None, white_bg=True,
                is_train=False, ndc_ray=False, N_samples=-1, tonemap=True):
        # rays_chunk: (N, (origin, viewdir, ray_up))

        # sample points
        device = rays_chunk.device

        # xyz_sampled2, ray_valid2, max_samps2, z_vals2, dists2 = self.sampler2.sample(rays_chunk, focal, ndc_ray, override_near=override_near, is_train=is_train, N_samples=N_samples)
        xyz_sampled, ray_valid, max_samps, z_vals, dists, whole_valid = self.sampler.sample(rays_chunk, focal, ndc_ray, override_near=override_near, is_train=is_train, N_samples=N_samples)
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

        # weight[xyz_normed[..., 2] > 0.2] = 0

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

        # app stands for appearance
        pweight = weight[ray_valid]
        app_mask = (weight > self.rayMarch_weight_thres)
        papp_mask = app_mask[ray_valid]

        # debug = torch.zeros((B, n_samples, 3), dtype=torch.short, device=device)
        debug = torch.zeros((B, n_samples, 3), dtype=torch.float, device=device, requires_grad=False)
        bounce_count = 0

        rgb = torch.zeros((*full_shape[:2], 3), device=device, dtype=weight.dtype)

        if app_mask.any():
            #  Compute normals for app mask

            norms = self.calculate_normals(xyz_sampled[papp_mask])
            world_normal[papp_mask] = norms
            # pred norms is initialized to world norms to set loss to zero for align_world_loss when prediction is none
            p_world_normal[papp_mask] = norms.detach()

            app_xyz = xyz_normed[papp_mask]

            if all_app_features is None:
                app_features = self.rf.compute_appfeature(app_xyz)
            else:
                # app_features = all_app_features[papp_mask]
                _, app_features = self.rf.compute_feature(app_xyz)

            # get base color of the point
            diffuse, tint, matprop = self.diffuse_module(
                app_xyz, viewdirs[app_mask], app_features)
            # diffuse = diffuse.type(rgb.dtype)

            noise_app_features = (app_features + torch.randn_like(app_features) * self.appdim_noise_std)

            # interpolate between the predicted and world normals
            if self.normal_module is not None:
                p_world_normal[papp_mask] = self.normal_module(app_xyz, app_features)
                l = self.l
                v_world_normal = ((1-l)*p_world_normal + l*world_normal)
                v_world_normal = v_world_normal / (v_world_normal.norm(dim=-1, keepdim=True) + 1e-8)
                # TODO REMOVE
                if FIXED_SPHERE:
                    v_world_normal = xyz_sampled[..., :3] / (xyz_sampled[..., :3].norm(dim=-1, keepdim=True) + 1e-8)
                # v_world_normal = xyz_sampled[..., :3] / (xyz_sampled[..., :3].norm(dim=-1, keepdim=True) + 1e-8)
                # world_normal = xyz_sampled[..., :3] / (xyz_sampled[..., :3].norm(dim=-1, keepdim=True) + 1e-8)
            else:
                v_world_normal = world_normal

            # calculate reflected ray direction
            V = -viewdirs[app_mask]
            L = v_world_normal[papp_mask]
            VdotL = (V * L).sum(-1, keepdim=True)
            refdirs = 2 * VdotL * L - V

            reflect_rgb = torch.zeros_like(diffuse)
            roughness = matprop['roughness'].squeeze(-1)
            # roughness = torch.where((xyz_sampled[..., 0].abs() < 0.15) | (xyz_sampled[..., 1].abs() < 0.15), 0.30, 0.15)[papp_mask]
            # roughness = 1e-2*torch.ones_like(roughness)
            if recur >= self.max_recurs and self.ref_module is None:
                ratio_diffuse = 1
                ratio_reflected = 0
            elif self.ref_module is not None and recur >= self.max_recurs:
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
                ratio_diffuse = matprop['ratio_diffuse']
                ratio_reflected = 1 - ratio_diffuse
                bounce_mask, full_bounce_mask, inv_full_bounce_mask, ray_mask = self.selector(
                        app_mask, weight.detach(), VdotL, 1-roughness.detach(), num_roughness_rays)
                # if the bounce is not calculated, set the ratio to 0 to make sure we don't get black spots
                # if not bounce_mask.all() and not is_train:
                #     ratio_diffuse[~bounce_mask] += ratio_reflected[~bounce_mask]
                #     ratio_reflected[~bounce_mask] = 0
                # if not is_train:
                # ic(ray_mask.shape, ray_mask.sum(), bounce_mask.sum(), bounce_mask.shape)

                if bounce_mask.sum() > 0:
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
                    D = bounce_rays.shape[-1]

                    if recur == 0 and self.world_bounces > 0:
                        incoming_light = torch.zeros((bounce_rays.shape[0], bounce_rays.shape[1], 3), device=device)
                        # TODO update with ray mask
                        with torch.no_grad():
                            reflect_data = self(bounce_rays[:, :self.world_bounces, :].reshape(-1, D), focal, recur=recur+1, white_bg=False,
                                                override_near=0.15, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples, tonemap=False)


                        incoming_light[:, :self.world_bounces, :] = reflect_data['rgb_map'].reshape(-1, self.world_bounces, 3)
                        mipval = mipval.reshape(-1, num_roughness_rays)
                        # apply ray mask
                        incoming_light[:, self.world_bounces:, :] = self.render_just_bg(
                                bounce_rays[:, self.world_bounces:, :].reshape(-1, D),
                                mipval[:, self.world_bounces:].reshape(-1)
                            ).reshape(-1, num_roughness_rays-self.world_bounces, 3)
                    else:
                        incoming_light = self.render_just_bg(bounce_rays.reshape(-1, D), mipval.reshape(-1))

                    # self.brdf_sampler.update(bounce_rays[..., :3].reshape(-1, 3), mipval.reshape(-1), incoming_light.reshape(-1, 3))
                    # miplevel = self.bg_module.sa2mip(mipval)
                    # debug[full_bounce_mask][..., 0] += miplevel.mean(dim=1) / (self.bg_module.max_mip-1)
                    
                    tinted_ref_rgb = self.brdf(incoming_light, V[bounce_mask], bounce_rays[..., 3:6], outward.reshape(-1, 1, 3), app_features[bounce_mask], roughness[bounce_mask], matprop, bounce_mask, ray_mask)
                    s = row_mask_sum(incoming_light, ray_mask) / (ray_mask.sum(dim=1)+1e-8)[..., None]
                    # s = incoming_light.max(dim=1).values

                    # if not is_train:
                    #     plt.style.use('dark_background')
                    #     plt.scatter(self.brdf_sampler.angs[:100, 0], self.brdf_sampler.angs[:100, 1], c=incoming_light.clip(0, 1).detach().cpu()[0])
                    #     plt.show()

                    # s = incoming_light[:, 0]
                    debug[full_bounce_mask] += s# / (s+1)
                    # debug[full_bounce_mask] += bounce_rays[:, 0, 3:6]/2 + 0.5
                    reflect_rgb[bounce_mask] = tint[bounce_mask] * tinted_ref_rgb
                    # reflect_rgb[bounce_mask] = incoming_light.mean(dim=1)
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
            # reflectivity = matprop['reflectivity']
            # rgb[app_mask] = tint * ((1-reflectivity)*matprop['ambient'] + reflectivity * reflect_rgb)
            # rgb[app_mask] = reflect_rgb + matprop['diffuse']
            rgb[app_mask] = reflect_rgb# + matprop['diffuse']
            # rgb[app_mask] = tint * reflectivity * reflect_rgb + (1-reflectivity)*matprop['diffuse']
            # rgb[app_mask] = tint * (ambient + reflectivity * reflect_rgb)

            align_world_loss = (1-(p_world_normal * world_normal).sum(dim=-1).clamp(max=self.max_normal_similarity))
            # align_world_loss = torch.linalg.norm(p_world_normal - world_normal, dim=-1)
            normal_loss = (pweight * align_world_loss).sum() / B
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
        backwards_rays_loss = torch.matmul(viewdirs[ray_valid].reshape(-1, 1, 3), p_world_normal.reshape(-1, 3, 1)).reshape(pweight.shape).clamp(min=0)**2
        backwards_rays_loss = (pweight * backwards_rays_loss).sum() / pweight.sum().clip(min=1e-10)

        # calculate depth

        # shadow_map = torch.sum(weight * shadows, 1)
        # (N, bundle_size, bundle_size)
        acc_map = torch.sum(weight, 1)
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
            # p_world_normal_map = torch.sum(weight[..., None] * p_world_normal, 1)
            # p_world_normal_map = p_world_normal_map / \
            #     (torch.norm(p_world_normal_map, dim=-1, keepdim=True)+1e-8)
            # d_world_normal_map = torch.sum(weight[..., None] * world_normal, 1)
            # d_world_normal_map = d_world_normal_map / (torch.linalg.norm(d_world_normal_map, dim=-1, keepdim=True)+1e-8)
            # full_v_world_normal = torch.zeros(full_shape, device=device)
            # full_v_world_normal[ray_valid] = v_world_normal
            # v_world_normal_map = torch.sum(weight[..., None] * full_v_world_normal, 1)
            v_world_normal_map = row_mask_sum(pweight[..., None] * v_world_normal, ray_valid)
            # v_world_normal_map = v_world_normal_map / (torch.linalg.norm(v_world_normal_map, dim=-1, keepdim=True)+1e-8)
            # d_normal_map = torch.matmul(row_basis, d_world_normal_map.unsqueeze(-1)).squeeze(-1)
            # p_normal_map = torch.matmul(
            #     row_basis, p_world_normal_map.unsqueeze(-1)).squeeze(-1)
            # v_normal_map = torch.matmul(row_basis, v_world_normal_map.unsqueeze(-1)).squeeze(-1)
            # v_normal_map = v_normal_map / (torch.linalg.norm(d_normal_map, dim=-1, keepdim=True)+1e-8)
            v_world_normal_map = acc_map[..., None] * v_world_normal_map + (1 - acc_map[..., None])

            if weight.shape[1] > 0:
                # inds = ((weight * (alpha < self.alphaMask_thres)).max(dim=1).indices).clip(min=0)
                inds = ((weight).max(dim=1).indices).clip(min=0)
                full_xyz_sampled = torch.zeros((B, max_samps, 4), device=device)
                full_xyz_sampled[ray_valid] = xyz_sampled

                # w1 = (1-torch.linalg.norm(full_xyz_sampled[..., :3], dim=-1)).abs()
                # w2 = torch.where((full_xyz_sampled[..., :3] * full_xyz_sampled[range(full_shape[0]), inds][:, None, :3]).sum(dim=-1) > 0.9, 1, 9999999999)
                # inds = (w1*w2).min(dim=1).indices
                # ic((w1*w2).min(dim=1).values[acc_map > 0.5].mean(), (w1).min(dim=1).values[acc_map > 0.5].mean())
                termination_xyz = full_xyz_sampled[range(full_shape[0]), inds].cpu()
            else:
                termination_xyz = torch.empty(0, 4)
            # ic((1-torch.linalg.norm(full_xyz_sampled[..., :3], dim=-1)).abs().min(dim=1).values[acc_map > 0.5].mean())
            # debug_map = (1-torch.linalg.norm(termination_xyz[..., :3], dim=-1, keepdim=True)).abs()
            # debug_map = debug_map.expand(-1, 3)*500
            v_world_normal_map = termination_xyz[..., :3] / torch.linalg.norm(termination_xyz[..., :3], dim=-1, keepdim=True)

            # collect statistics about the surface
            # surface width in voxels
            surface_width = app_mask.sum(dim=1)

            # i = torch.where(acc_map > 0.5)[0][0]
            # weight_slice = weight[i].reshape(-1).cpu()
            # print(z_vals[i])
            # print(weight_slice, surface_width[i])
            # plt.bar(z_vals[i].cpu(), weight_slice)
            # plt.show()

            # TODO REMOVE
            LOGGER.log_norms_n_rays(xyz_sampled[papp_mask], v_world_normal[papp_mask], weight[app_mask])
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if tonemap:
            rgb_map = self.tonemap(rgb_map.clip(min=0), noclip=True)
        # ic(weight.mean(), rgb.mean(), rgb_map.mean(), v_world_normal.mean(), sigma.mean(), dists.mean(), alpha.mean())

        if self.bg_module is not None and not white_bg:
            bg_roughness = torch.zeros(B, 1, device=device)
            bg = self.bg_module(viewdirs[:, 0, :], bg_roughness)
            rgb_map = rgb_map + \
                (1 - acc_map[..., None]) * self.tonemap(bg.reshape(-1, 3), noclip=True)
        else:
            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1 - acc_map[..., None])

        debug_map = (weight[..., None]*debug).sum(dim=1)
        # debug_map = ray_valid.sum(dim=1, keepdim=True).expand(-1, 3) / 256

        return dict(
            rgb_map=rgb_map,
            depth_map=depth_map.detach().cpu(),
            debug_map=debug_map.detach().cpu(),
            normal_map=v_world_normal_map.detach().cpu(),
            # weight_slice=weight_slice,
            recur=recur,
            acc_map=acc_map.detach().cpu(),
            roughness=roughness.mean(),
            diffuse_reg=roughness.mean() + diffuse.mean(),# + ((tint_brightness-0.5)**2).mean(),
            normal_loss=normal_loss,
            backwards_rays_loss=backwards_rays_loss,
            termination_xyz=termination_xyz,
            floater_loss=floater_loss,
            surf_width=surface_width,
            color_count=app_mask.detach().sum(),
            bounce_count=bounce_count,
            whole_valid=whole_valid, 
        )
