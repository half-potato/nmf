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
import math
from .logger import Logger

LOGGER = Logger(enable=False)

def snells_law(r, n, l):
    # n: (B, 3) surface outward normal
    # l: (B, 3) light direction towards surface
    # r: ratio between indices of refraction. n1/n2
    # where n1 = index where light starts and n2 = index after surface penetration
    dtype = n.dtype
    n = n.double()
    l = l.double()

#     Vec3f refract(const Vec3f &I, const Vec3f &N, const float &ior) 
# { 
#     float cosi = clamp(-1, 1, dotProduct(I, N)); 
#     float etai = 1, etat = ior; 
#     Vec3f n = N; 
#     if (cosi < 0) { cosi = -cosi; } else { std::swap(etai, etat); n= -N; } 
#     float eta = etai / etat; 
#     float k = 1 - eta * eta * (1 - cosi * cosi); 
#     return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n; 
# }

    cosi = torch.matmul(n.reshape(-1, 1, 3), l.reshape(-1, 3, 1)).reshape(*n.shape[:-1], 1)
    Nsign = torch.sign(cosi)
    N = torch.where(cosi < 0, n, -n)
    cosi = cosi * Nsign
    R = torch.where(cosi < 0, 1/r, r)

    k = 1 - R * R * (1 - cosi * cosi);
    refractdir = R * l + (R * cosi - torch.sqrt(k.clip(min=0))) * N

    # c = -torch.matmul(n.reshape(-1, 1, 3), l.reshape(-1, 3, 1)).reshape(*n.shape[:-1], 1)
    # sign = torch.sign(c).abs()
    # refractdir = (r*l + (r * c.abs() - torch.sqrt( (1 - r**2 * (1-c**2)).clip(min=1e-8) )) * sign*n)
    return refractdir.type(dtype)

def fresnel_law(ior1, ior2, n, l, o):
    # input: 
    #  n: (B, 3) surface outward normal
    #  l: (B, 3) light direction towards surface
    #  o: (B, 3) refracted light direction given by snells_law
    #  ior1: index of refraction for material from which light was emitted
    #  ior2: index of refraction for material after surface
    # output:
    #  ratio reflected, between 0 and 1
    cos_i = torch.matmul(n.reshape(-1, 1, 3), l.reshape(-1, 3, 1)).reshape(*n.shape[:-1], 1)
    cos_t = torch.matmul(n.reshape(-1, 1, 3), o.reshape(-1, 3, 1)).reshape(*n.shape[:-1], 1)
    sin_t = torch.sqrt(1 - cos_t**2)
    s_polar = (ior2 * cos_i - ior1 * cos_t) / (ior2 * cos_i + ior1 * cos_t)
    p_polar = (ior2 * cos_t - ior1 * cos_i) / (ior2 * cos_t + ior1 * cos_i)
    ratio_reflected = (s_polar + p_polar)/2
    return torch.where(sin_t >= 1, torch.ones_like(ratio_reflected), ratio_reflected)

def refract_reflect(ior1, ior2, n, l, p):
    # n: (B, 3) surface outward normal
    # l: (B, 3) light direction towards surface
    # p: (B) reflectivity of material, between 0 and 1
    # ior1: index of refraction for material from which light was emitted
    # ior2: index of refraction for material after surface
    ratio = ior2/ior1
    o = snells_law(ratio, n, l)
    ratio_reflected = fresnel_law(ior1, ior2, n, l, o)
    ratio_refracted = 1 - ratio_reflected
    out_ratio_reflected = 1 - p * ratio_refracted
    return out_ratio_reflected



def select_top_n_app_mask(app_mask, weight, prob, N, t=0, wt=0):
    # weight: (B, N)
    # prob: (M)
    # app_mask: (B, N) with M true elements
    # N: int max number of selected
    # t: threshold

    pweight = weight[app_mask]
    # topmask = weight > (weight.max(dim=1, keepdim=True).values - wt)
    # ptopmask = topmask[app_mask]
    prob = prob.reshape(-1)# * ptopmask
    M = prob.shape[0]
    bounce_mask = torch.zeros((M), dtype=bool, device=app_mask.device)
    full_bounce_mask = torch.zeros_like(app_mask)
    inv_full_bounce_mask = torch.zeros_like(app_mask)

    n_bounces = min(min(N, M), ((prob > t) & (pweight > wt)).sum())
    inds = torch.argsort(-pweight*prob)[:n_bounces]
    bounce_mask[inds] = 1

    # combine the two masks because double masking causes issues
    ainds, ajinds = torch.where(app_mask)
    full_bounce_mask[ainds[bounce_mask], ajinds[bounce_mask]] = 1
    inv_full_bounce_mask[ainds[~bounce_mask], ajinds[~bounce_mask]] = 1
    return bounce_mask, full_bounce_mask, inv_full_bounce_mask

def raw2alpha(sigma, flip, dist):
    # sigma, dist  [N_rays, N_samples]
    v = torch.exp(-sigma*dist)
    alpha = torch.where(flip, v, 1-v)
    ic(alpha.max())
    # alpha = 1. - torch.exp(-sigma*dist)

    # T is the term that integrates the alpha backwards to prevent occluded objects from influencing things
    # multiply in exponential space to take exponential of integral
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

    def contract_coord(self, xyz_sampled): 
        dist = torch.linalg.norm(xyz_sampled[..., :3], dim=1, keepdim=True) + 1e-8
        direction = xyz_sampled[..., :3] / dist
        contracted = torch.where(dist > 1, (2-1/dist), dist) * direction
        return torch.cat([ contracted, xyz_sampled[..., 3:] ], dim=-1)


class TensorNeRF(torch.nn.Module):
    def __init__(self, rf, grid_size, aabb, diffuse_module, normal_module=None, ref_module=None, bg_module=None,
                 alphaMask=None, near_far=[2.0, 6.0], nEnvSamples=100, specularity_threshold=0.005, max_recurs=0,
                 max_normal_similarity=1, infinity_border=False, min_refraction=1.1, enable_refraction=True,
                 density_shift=-10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                 max_bounce_rays=4000, roughness_rays=3, bounce_min_weight=0.001,
                 fea2denseAct='softplus', enable_alpha_mask=True):
        super(TensorNeRF, self).__init__()
        self.rf = rf(aabb=aabb, grid_size=grid_size)
        self.ref_module = ref_module(in_channels=self.rf.app_dim) if ref_module is not None else None
        self.normal_module = normal_module(in_channels=self.rf.app_dim) if normal_module is not None else None
        self.diffuse_module = diffuse_module(in_channels=self.rf.app_dim)
        self.bg_module = bg_module

        self.alphaMask = alphaMask
        self.infinity_border = infinity_border
        self.enable_alpha_mask = enable_alpha_mask

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct


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

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001, lr_bg=0.03):
        grad_vars = []
        grad_vars += self.rf.get_optparam_groups(lr_init_spatial, lr_init_network)
        if self.ref_module is not None:
            grad_vars += [{'params': list(self.ref_module.parameters()), 'lr': 1e-3}]
        if hasattr(self, 'normal_module') and isinstance(self.normal_module, torch.nn.Module):
            grad_vars += [{'params': self.normal_module.parameters(),
                           'lr': lr_init_network}]
        if hasattr(self, 'diffuse_module') and isinstance(self.diffuse_module, torch.nn.Module):
            grad_vars += [{'params': self.diffuse_module.parameters(),
                           'lr': lr_init_network}]
        if hasattr(self, 'bg_module') and isinstance(self.bg_module, torch.nn.Module):
            grad_vars += [{'params': self.bg_module.parameters(),
                'lr': lr_bg, 'name': 'bg'}]
        return grad_vars

    def save(self, path, config):
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
    def load(ckpt):
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
            rf.alphaMask = AlphaGridMask(
                ckpt['alphaMask.aabb'], alpha_volume)
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
                    xyz_sampled).view(xyz_sampled.shape[:-1]) > self.alphaMask_thres, self.rf.contract_space).any(-1)

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

    def recover_envmap(self, res, xyz=None):
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

        if self.ref_module is not None:
        # roughness = 1/np.pi*torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            roughness = 20 * torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            envmap = self.ref_module(xyz_samp, staticdir, app_features, refdirs=ang_vecs.reshape(
                -1, 3), roughness=roughness).reshape(res, 2*res, 3)
        else:
            envmap = torch.zeros(res, 2*res, 3)
        color, tint, _, _, _, _ = self.diffuse_module(xyz_samp, ang_vecs.reshape(-1, 3), app_features)
        color = color.reshape(res, 2*res, 3)
        
        return envmap, color

    def at_infinity(self, xyz_sampled, max_dist=10):
        margin = 1 - 1/max_dist/2
        at_infinity = torch.linalg.norm(
            xyz_sampled, dim=-1, ord=torch.inf).abs() >= margin
        return at_infinity

    def roughness2noisestd(self, roughness):
        slope = 6
        zero_point = 0.001
        offset = math.exp(slope*(zero_point-1))
        return torch.exp(slope*(roughness-1)).clip(min=0, max=1) - offset

    def calculate_normals(self, xyz):
        with torch.enable_grad():
            xyz_g = xyz.clone()
            if not xyz_g.requires_grad:
                xyz_g.requires_grad = True

            # compute sigma
            xyz_g_normed = self.rf.normalize_coord(xyz_g)
            sigma_feature = self.rf.compute_densityfeature(xyz_g_normed)
            validsigma = self.feature2density(sigma_feature)

            # compute normal
            grad_outputs = torch.ones_like(validsigma)
            g = grad(validsigma, xyz_g, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)
            norms = -g[0][:, :3]
            norms = norms / (torch.linalg.norm(norms, dim=-1, keepdim=True) + 1e-8)
            return norms

    def forward(self, rays_chunk, focal, recur=0, init_refraction_index=torch.tensor(1.0), override_near=None, output_alpha=None, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
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
        rays_o = rays_chunk[:, :3]
        rays_up = rays_chunk[:, 6:9]
        rays_up = rays_up.view(-1, 1, 3).expand(xyz_sampled_shape)
        B = xyz_sampled.shape[0]
        n_samples = xyz_sampled_shape[1]

        ior_i_full = init_refraction_index.to(device).reshape(-1, 1).expand(xyz_sampled_shape[:-1])
        flip = ior_i_full > self.min_refraction

        # sample alphas and cull samples from the ray
        alphas = torch.zeros(xyz_sampled_shape[:-1], device=device)
        if self.alphaMask is not None and self.enable_alpha_mask and not flip.any():
            alphas[ray_valid] = self.alphaMask.sample_alpha(
                xyz_sampled[ray_valid], contract_space=self.rf.contract_space)

            # T = torch.cumprod(torch.cat([
            #     torch.ones(alphas.shape[0], 1, device=alphas.device),
            #     1. - alphas + 1e-10
            # ], dim=-1), dim=-1)[:, :-1]
            
            alpha_mask = (alphas > self.alphaMask_thres)# & (T > 0)
            #  ic((T <= 0).sum())
            ray_invalid = ~ray_valid
            ray_invalid |= (~alpha_mask)
            ray_valid = ~ray_invalid

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
            ic(validsigma.max(), sigma_feature.max())

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

        floater_loss = torch.einsum('...j,...k->...', weight.reshape(B, -1), weight.reshape(B, -1)).mean()

        # app stands for appearance
        app_mask = (weight > self.rayMarch_weight_thres)
        ic(app_mask.sum(), weight.max())

        # debug = torch.zeros((B, n_samples, 3), dtype=torch.short, device=device)
        with torch.no_grad():
            debug = torch.zeros((B, n_samples, 3), dtype=torch.float, device=device)
            recur_depth = z_vals.clone()
            depth_map = torch.sum(weight * recur_depth, 1)
            acc_map = torch.sum(weight, 1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        if app_mask.any():
            #  Compute normals for app mask

            norms = self.calculate_normals(xyz_sampled[app_mask])
            world_normal[app_mask] = norms
            # pred norms is initialized to world norms to set loss to zero for align_world_loss when prediction is none
            p_world_normal[app_mask] = norms.detach()

            app_features = self.rf.compute_appfeature(xyz_normed[app_mask])

            # interpolate between the predicted and world normals
            if self.normal_module is not None:
                p_world_normal[app_mask] = self.normal_module(xyz_normed[app_mask], app_features)
                l = self.l# if is_train else 1
                v_world_normal = (1-l)*p_world_normal + l*world_normal
                v_world_normal = v_world_normal / (v_world_normal.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                v_world_normal = world_normal

            # get base color of the point
            diffuse, tint, roughness, refraction_index, reflectivity, ratio_diffuse = self.diffuse_module(
                xyz_normed[app_mask], viewdirs[app_mask], app_features)
            diffuse = diffuse.type(rgb.dtype)

            # calculate reflected ray direction
            refdirs = viewdirs - 2 * \
                (viewdirs * v_world_normal).sum(-1, keepdim=True) * v_world_normal
            cosref = torch.matmul(refdirs.reshape(-1, 1, 3), v_world_normal.reshape(-1, 3, 1)).reshape(*viewdirs.shape[:-1], 1)

            ior_i = ior_i_full[app_mask].reshape(-1, 1)
            ior_t = refraction_index

            # calculated interpolated snell's law
            n = v_world_normal[app_mask].double()
            l = viewdirs[app_mask].double()

            # snell's law
            cos_i = torch.matmul(n.reshape(-1, 1, 3), l.reshape(-1, 3, 1)).reshape(*n.shape[:-1], 1)
            N = torch.where(cos_i < 0, n, -n)
            density_ratio = (ior_i/ior_t).double()
            R = torch.where(cos_i < 0, density_ratio, 1/density_ratio)
            cos_i = cos_i.abs()

            k = 1 - R * R * (1 - cos_i * cos_i)
            refractdirs = R * l + (R * cos_i - torch.sqrt(k.clip(min=1e-5))) * N
            refractdirs = refractdirs.float()
            refractdirs = refractdirs / (torch.linalg.norm(refractdirs, dim=1, keepdim=True)+1e-8)

            refract_sim = (refractdirs * l.float()).sum(dim=-1, keepdim=True)
            sim_weight = (1-refract_sim)/2

            # compute fresnel_law
            sin_t = R * torch.sqrt((1-cos_i**2).clip(min=1e-5))
            cos_t = torch.sqrt((1-sin_t**2).clip(min=1e-5))
            s_polar = (ior_t * cos_i - ior_i * cos_t) / (ior_t * cos_i + ior_i * cos_t)
            p_polar = (ior_i * cos_i - ior_t * cos_t) / (ior_t * cos_t + ior_i * cos_i)
            ratio_reflected = (s_polar**2 + p_polar**2)/2
            ratio_reflected = torch.where((sin_t >= 1), torch.ones_like(ratio_reflected), ratio_reflected)

            # beyond a certain roughness, it's probably better to try and do view dependent color
            r = (1-reflectivity) * (1 - ratio_reflected).float()
            ratio_refracted = ((1-ratio_diffuse) * r)
            ratio_reflected = ((1-ratio_diffuse) * (1 - r))

            refract_rgb = torch.zeros_like(diffuse)
            reflect_rgb = torch.zeros_like(diffuse)

            M = diffuse.shape[0]
            if recur >= self.max_recurs:
                ratio_diffuse = 1
                ratio_reflected = 0
                ratio_refracted = 0
            else:
                num_roughness_rays = 1 if recur > 0 else self.roughness_rays
                # compute which rays to refract
                rbounce_mask, rfull_bounce_mask, rinv_full_bounce_mask = select_top_n_app_mask(
                        app_mask, weight, ratio_refracted*(refraction_index > self.min_refraction), self.max_bounce_rays,
                        0.01, self.bounce_min_weight)
                if not rbounce_mask.all() and not is_train:
                    ratio_diffuse[~rbounce_mask] += ratio_refracted[~rbounce_mask]
                    ratio_refracted[~rbounce_mask] = 0

                if rbounce_mask.sum() > 0:
                    # decide how many bounces to calculate
                    rbounce_rays = torch.cat([
                        xyz_sampled[rfull_bounce_mask][..., :3],
                        refractdirs[rbounce_mask],
                        rays_up[rfull_bounce_mask]
                    ], dim=-1)

                    # add noise to simulate roughness
                    D = rbounce_rays.shape[-1]
                    ray_noise = torch.normal(0, 1, (rbounce_rays.shape[0], num_roughness_rays, 3), device=device) * self.roughness2noisestd(roughness[rbounce_mask].reshape(-1, 1, 1))
                    rbounce_rays = rbounce_rays.reshape(-1, 1, D).repeat(1, num_roughness_rays, 1)
                    ray_noise[:, 0] = 0
                    noise_rays = rbounce_rays[..., 3:6] + ray_noise
                    rbounce_rays[..., 3:6] = noise_rays / (torch.linalg.norm(noise_rays, dim=-1, keepdim=True)+1e-8)

                    rinit_refraction_index = refraction_index[rbounce_mask]
                    rinit_refraction_index = torch.where(flip[app_mask][rbounce_mask][:, None], torch.ones_like(rinit_refraction_index), rinit_refraction_index)
                    rinit_refraction_index = rinit_refraction_index.reshape(-1, 1).repeat(1, num_roughness_rays).reshape(-1)
                    refract_data = self(rbounce_rays.reshape(-1, D), focal, recur=recur+1, init_refraction_index=rinit_refraction_index, white_bg=white_bg,
                                    override_near=0.15, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples)

                    refract_depth = refract_data['depth_map'].reshape(-1, num_roughness_rays)
                    refract_col = refract_data['rgb_map'].reshape(-1, num_roughness_rays, 3)
                    recur_depth[rfull_bounce_mask] = refract_depth.max(dim=1).values

                    refract_col = refract_col.mean(dim=1)
                    m = rfull_bounce_mask.sum(dim=1) > 0
                    LOGGER.log_rays(rays_chunk[m].reshape(-1, D), recur, dict(depth_map=depth_map.detach()[m]))
                    LOGGER.log_rays(rbounce_rays.reshape(-1, D), recur+1, refract_data)
                    # recur += refract_data['recur']

                    refract_rgb[rbounce_mask] = (diffuse[rbounce_mask] * refract_col)
                    debug[rfull_bounce_mask] += ratio_refracted[rbounce_mask] * diffuse[rbounce_mask] * refract_col


                # compute which rays to reflect
                bounce_mask, full_bounce_mask, inv_full_bounce_mask = select_top_n_app_mask(
                        app_mask, weight, ratio_reflected, self.max_bounce_rays,
                        self.specularity_threshold, self.bounce_min_weight)
                # if the bounce is not calculated, set the ratio to 0 to make sure we don't get black spots
                if not bounce_mask.all() and not is_train:
                    ratio_diffuse[~bounce_mask] += ratio_reflected[~bounce_mask]
                    ratio_reflected[~bounce_mask] = 0

                if bounce_mask.sum() > 0:
                    # decide how many bounces to calculate
                    bounce_rays = torch.cat([
                        xyz_sampled[full_bounce_mask][..., :3] + 1e-5 * refdirs[full_bounce_mask],
                        refdirs[full_bounce_mask],
                        rays_up[full_bounce_mask]
                    ], dim=-1)

                    # add noise to simulate roughness
                    D = bounce_rays.shape[-1]
                    ray_noise = torch.normal(0, 1, (bounce_rays.shape[0], num_roughness_rays, 3), device=device) * self.roughness2noisestd(roughness[bounce_mask].reshape(-1, 1, 1))
                    bounce_rays = bounce_rays.reshape(-1, 1, D).repeat(1, num_roughness_rays, 1)
                    noise_rays = bounce_rays[..., 3:6] + ray_noise
                    bounce_rays[..., 3:6] = noise_rays / (torch.linalg.norm(noise_rays, dim=-1, keepdim=True)+1e-8)

                    reflect_data = self(bounce_rays.reshape(-1, D), focal, recur=recur+1, white_bg=white_bg,
                                    override_near=0.15, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples)

                    ref_rgb = reflect_data['rgb_map'].reshape(-1, num_roughness_rays, 3).mean(dim=1)
                    tinted_ref_rgb = (diffuse[bounce_mask] * ref_rgb).clip(min=0, max=1)
                    # TODO uncomment?
                    # debug[full_bounce_mask] += ratio_reflected[bounce_mask] * tinted_ref_rgb
                    reflect_rgb[bounce_mask] = tinted_ref_rgb
                    # recur += reflect_data['recur']

                    m = full_bounce_mask.sum(dim=1) > 0
                    # LOGGER.log_rays(rays_chunk[m].reshape(-1, D), recur, dict(depth_map=depth_map.detach()[m]))
                    # LOGGER.log_rays(bounce_rays.reshape(-1, D), recur+1, reflect_data)

                # elif self.ref_module is not None and inv_full_bounce_mask.any():
                #     # compute other reflections using ref module
                #     ref_rgb = self.ref_module(
                #         xyz_normed[inv_full_bounce_mask], viewdirs[inv_full_bounce_mask],
                #         app_features[~bounce_mask], refdirs=refdirs[inv_full_bounce_mask],
                #         roughness=roughness[~bounce_mask])
                #     tinted_ref_rgb = (1-roughness[~bounce_mask]) * tint[~bounce_mask] * reflect_data['rgb_map']
                #     rgb[inv_full_bounce_mask] = tinted_ref_rgb
            # elif self.ref_module is not None:
            #     ref_rgbs = self.ref_module(
            #         xyz_normed[app_mask], viewdirs[app_mask],
            #         app_features, refdirs=refdirs[app_mask],
            #         roughness=roughness)#.detach()
            #     rgb[app_mask] += (tint * ref_rgbs).type(rgb.dtype)
            rgb[app_mask] = ratio_diffuse * diffuse + ratio_reflected * reflect_rgb + ratio_refracted * refract_rgb
            # if recur == 1 and init_refraction_index.mean() > 1.1:
            #     bad_mask = (rgb.mean(dim=-1) > 0.5) & (rgb.mean(dim=-1) < 0.8)
            #     ic(torch.where(bad_mask))
            #     ic(weight[bad_mask].mean(), weight[:, 91].mean())
            # if recur == 0:
            #     ic(torch.histogram(alpha.detach().cpu(), bins=torch.linspace(0, 1, 10)))

            align_world_loss = -(p_world_normal[app_mask] * world_normal[app_mask]).sum(dim=-1).clamp(max=self.max_normal_similarity)
            normal_loss = (weight[app_mask] * (align_world_loss)).mean()
        else:
            ratio_diffuse = torch.tensor(0.0)
            v_world_normal = world_normal
            roughness = torch.tensor(0.0)
            normal_loss = torch.tensor(0.0)
        
        backwards_rays_loss = -torch.matmul(viewdirs.reshape(-1, 1, 3), v_world_normal.reshape(-1, 3, 1)).reshape(app_mask.shape).clamp(max=0)
        backwards_rays_loss = (weight * (backwards_rays_loss)).mean()

        # calculate depth

        # shadow_map = torch.sum(weight * shadows, 1)
        # (N, bundle_size, bundle_size)
        with torch.no_grad():
            depth_map = torch.sum(weight * recur_depth, 1)
            acc_map = torch.sum(weight, 1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

            # view dependent normal map
            # N, 3, 3
            row_basis = torch.stack([
                torch.cross(viewdirs[:, 0], rays_up[:, 0]),
                viewdirs[:, 0],
                -rays_up[:, 0],
            ], dim=1)
            p_world_normal_map = torch.sum(weight[..., None] * p_world_normal, 1)
            p_world_normal_map = p_world_normal_map / \
                (torch.norm(p_world_normal_map, dim=-1, keepdim=True)+1e-8)
            d_world_normal_map = torch.sum(weight[..., None] * world_normal, 1)
            d_world_normal_map = d_world_normal_map / (torch.linalg.norm(d_world_normal_map, dim=-1, keepdim=True)+1e-8)
            v_world_normal_map = torch.sum(weight[..., None] * v_world_normal, 1)
            v_world_normal_map = v_world_normal_map / (torch.linalg.norm(d_world_normal_map, dim=-1, keepdim=True)+1e-8)
            d_normal_map = torch.matmul(row_basis, d_world_normal_map.unsqueeze(-1)).squeeze(-1)
            p_normal_map = torch.matmul(
                row_basis, p_world_normal_map.unsqueeze(-1)).squeeze(-1)
            v_normal_map = torch.matmul(row_basis, v_world_normal_map.unsqueeze(-1)).squeeze(-1)
            # v_normal_map = v_normal_map / (torch.linalg.norm(d_normal_map, dim=-1, keepdim=True)+1e-8)

            # # extract
            inds = ((weight * (alpha < self.alphaMask_thres)).max(dim=1).indices).clip(min=0)
            termination_xyz = xyz_sampled[range(xyz_sampled_shape[0]), inds]

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

        # process debug to turn it into map
        # 1 = refracted. -1 = reflected
        # we want to represent these two values by represnting them in the 1s and 10s place
        debug_map = (weight[..., None]*debug).sum(dim=1).clamp(0, 1)
        return dict(
            rgb_map=rgb_map,
            depth_map=depth_map,
            debug_map=debug_map,
            normal_map=v_normal_map.cpu(),
            # normal_map=v_world_normal_map.cpu(),
            recur=recur,
            acc_map=acc_map,
            normal_loss=normal_loss,
            backwards_rays_loss=backwards_rays_loss,
            termination_xyz=termination_xyz.cpu(),
            floater_loss=floater_loss,
            roughness=roughness.mean().cpu(),
        )
