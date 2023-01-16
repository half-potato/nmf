import torch
from modules.pt_selectors import select_bounces
from mutils import normalize
from modules.row_mask_sum import row_mask_sum
from icecream import ic
import math

class RealBounce(torch.nn.Module):
    def __init__(self, app_dim, diffuse_module, brdf, brdf_sampler, 
                 anoise, max_brdf_rays, percent_bright, cold_start_bg_iters, detach_N_iters, bright_sampler=None):
        super().__init__()
        self.diffuse_module = diffuse_module(in_channels=app_dim)
        self.brdf = brdf(in_channels=app_dim)
        self.brdf_sampler = brdf_sampler(max_samples=1024)
        self.bright_sampler = bright_sampler if bright_sampler is None else bright_sampler(max_samples=int(100*percent_bright+1))

        self.anoise = anoise
        self.max_brdf_rays = max_brdf_rays
        self.percent_bright = percent_bright
        self.cold_start_bg_iters = cold_start_bg_iters
        self.detach_bg = True
        self.detach_N_iters = detach_N_iters
        self.detach_N = True
        self.outputs = {'diffuse': 3, 'roughness': 1, 'tint': 3, 'spec': 3}

    def get_optparam_groups(self, lr_scale=1):
        grad_vars = []
        grad_vars += [{'params': self.diffuse_module.parameters(),
                       'lr': self.diffuse_module.lr*lr_scale}]
        grad_vars += [{'params': self.brdf.parameters(),
                       'lr': self.brdf.lr*lr_scale}]
        return grad_vars

    def check_schedule(self, iter, batch_mul, bg_module, **kwargs):
        if self.bright_sampler is not None:
            self.bright_sampler.check_schedule(iter, batch_mul, bg_module)
        if iter > batch_mul*self.cold_start_bg_iters:
            self.detach_bg = False
        if iter > batch_mul*self.detach_N_iters:
            self.detach_N = False
        return False

    @torch.no_grad()
    def graph_brdfs(self, xyzs, viewdirs, app_features, res):
        device = app_features.device

        # incoming light directions
        ele_grid, azi_grid = torch.meshgrid(
            torch.linspace(-math.pi/2, math.pi/2, res, dtype=torch.float32),
            torch.linspace(0, 2*math.pi, 2*res, dtype=torch.float32), indexing='ij')
        ang_vecs = torch.stack([
            -torch.sin(ele_grid),
            torch.cos(ele_grid) * torch.sin(azi_grid),
            torch.cos(ele_grid) * torch.cos(azi_grid),
        ], dim=-1).reshape(1, 1, -1, 3).to(device)

        assert(xyzs.shape[0] == viewdirs.shape[0])
        assert(xyzs.shape[0] == app_features.shape[0])

        diffuse, tint, matprop = self.diffuse_module(
            xyzs, viewdirs, app_features)

        n_angs = ang_vecs.shape[-2]
        n_views = viewdirs.shape[0]
        n_features = app_features.shape[0]
        # N = n_features * n_views * n_angs

        L = ang_vecs.expand(n_features, n_views, n_angs, 3)
        eV = viewdirs.reshape(1, -1, 1, 3).expand(n_features, n_views, n_angs, 3)
        halfvec = normalize((L + eV)/2)
        r1 = matprop['r1'].reshape(-1, 1, 1).expand(n_features, n_views, n_angs).reshape(-1, 1)
        r2 = matprop['r2'].reshape(-1, 1, 1).expand(n_features, n_views, n_angs).reshape(-1, 1)
        efeatures = app_features.reshape(n_features, 1, 1, -1).expand(n_features, n_views, n_angs, app_features.shape[-1])
        eN = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, 1, 1, 3).expand(n_features, n_views, n_angs, 3)
        proportion = matprop['r1'].reshape(-1, 1, 1).expand(n_features, n_views, n_angs).reshape(-1, 1)

        # normal must be z up so we don't have to create a row basis and rotate everything
        diffvec = L

        brdf_weight = self.brdf(
            eV.reshape(-1, 3),
            L.reshape(-1, 3),
            eN.reshape(-1, 3),
            halfvec.reshape(-1, 3),
            diffvec.reshape(-1, 3),
            efeatures.reshape(-1, app_features.shape[-1]),
            r1.reshape(-1))

        # next, compute GGX distribution because it's not represented here
        brdf_colors = (self.brdf_sampler.compute_prob(halfvec, eN, r1, r2, proportion=proportion).reshape(-1, 1) * brdf_weight)

        # add indicator of view direction
        viewdir_ind = (L * eV).sum(dim=-1).reshape(n_features*n_views, n_angs).max(dim=1).indices
        brdf_colors = brdf_colors.reshape(n_features*n_views, n_angs, 3)
        brdf_colors[range(n_features*n_views), viewdir_ind, 0] = 0
        brdf_colors[range(n_features*n_views), viewdir_ind, 1] = 1
        brdf_colors[range(n_features*n_views), viewdir_ind, 2] = 0

        # reshape image
        brdf_colors = brdf_colors.reshape(n_features, n_views, res, 2*res, 3)
        im = brdf_colors.permute(0, 2, 1, 3, 4).reshape(n_features*res, 2*n_views*res, 3)
        return im



    def forward(self, xyzs, app_features, viewdirs, normals, weights, app_mask, B, recur, ray_cast_fn):
        # xyzs: (M, 4)
        # viewdirs: (M, 3)
        # normals: (M, 3)
        # weights: (M)
        # B: number of rays being cast
        # recur: recursion counter
        # ray_cast_fn: function that casts out rays to allow for recursion
        debug = {}

        noise_app_features = (app_features + torch.randn_like(app_features) * self.anoise)
        diffuse, tint, matprop = self.diffuse_module(
            xyzs, viewdirs, app_features)

        # pick rays to bounce
        num_brdf_rays = self.max_brdf_rays[recur] // B
        bounce_mask, ray_mask, bright_mask = select_bounces(
                weights, app_mask, num_brdf_rays, self.percent_bright)

        reflect_rgb = torch.zeros_like(diffuse)
        brdf_rgb = torch.zeros_like(diffuse)
        spec = torch.zeros_like(diffuse)

        ray_xyz = xyzs[bounce_mask][..., :3].reshape(-1, 1, 3).expand(-1, ray_mask.shape[1], 3)
        if bounce_mask.any() and ray_mask.any() and ray_xyz.shape[0] == ray_mask.shape[0]:
            bN = normals[bounce_mask]
            if self.detach_N:
                bN.detach_()
            bV = -viewdirs[bounce_mask]
            # r1 = matprop['r1'][bounce_mask]*0 + 0.0001
            # r2 = matprop['r2'][bounce_mask]*0 + 0.0001
            r1 = matprop['r1'][bounce_mask]
            r2 = matprop['r2'][bounce_mask]
            proportion = matprop['proportion'][bounce_mask]

            L, row_world_basis = self.brdf_sampler.sample(
                    bV, bN,
                    r1**2, r2**2, ray_mask, proportion=proportion)

            # Sample bright spots
            if self.bright_sampler is not None and self.bright_sampler.is_initialized():
                bL, bright_mask = self.bright_sampler.sample(bV, bN, ray_mask, bright_mask)
                pbright_mask = bright_mask[ray_mask]
                L[pbright_mask] = bL[bright_mask]

            n = ray_xyz.shape[0]
            m = ray_mask.shape[1]

            eV = bV.reshape(-1, 1, 3).expand(-1, m, 3)[ray_mask]
            eN = bN.reshape(-1, 1, 3).expand(-1, m, 3)[ray_mask]
            ea1 = r1.expand(ray_mask.shape)[ray_mask]
            ea2 = r2.expand(ray_mask.shape)[ray_mask]
            efeatures = noise_app_features[bounce_mask].reshape(n, 1, -1).expand(n, m, -1)[ray_mask]
            eproportion = proportion.expand(ray_mask.shape)[ray_mask]
            exyz = ray_xyz[ray_mask]

            # stacked = torch.cat([
            #     bV.reshape(-1, 1, 3).expand(-1, m, 3),
            #     bN.reshape(-1, 1, 3).expand(-1, m, 3),
            #     ray_xyz,
            #     r1.reshape(-1, 1, 1).expand(*ray_mask.shape, 1),
            #     noise_app_features[bounce_mask].reshape(n, 1, -1).expand(n, m, -1),
            # ], dim=-1)
            # masked = stacked[ray_mask]
            # eV = masked[:, 0:3]
            # eN = masked[:, 3:6]
            # exyz = masked[:, 6:9]
            # ea = masked[:, 9:10]
            # efeatures = masked[:, 10:]

            H = normalize((eV+L)/2)
            mipval = self.brdf_sampler.calculate_mipval(H.detach(), eV, eN.detach(), ray_mask, ea1.detach()**2, ea2.detach()**2, proportion=eproportion)

            bounce_rays = torch.cat([
                exyz + L*5e-3,
                L,
            ], dim=-1)
            n = bounce_rays.shape[0]
            D = bounce_rays.shape[-1]
            incoming_light = ray_cast_fn(bounce_rays, mipval)

            # calculate second part of BRDF
            n, m = ray_mask.shape
            diffvec = torch.matmul(row_world_basis.permute(0, 2, 1), L.unsqueeze(-1)).squeeze(-1)
            halfvec = torch.matmul(row_world_basis.permute(0, 2, 1), H.unsqueeze(-1)).squeeze(-1)
            # brdf_weight = self.brdf(eV, L, eN, halfvec, diffvec, efeatures, ea)
            brdf_weight = self.brdf(eV, L.detach(), eN.detach(), halfvec.detach(), diffvec.detach(), efeatures, ea1.detach())
            ray_count = (ray_mask.sum(dim=1)+1e-8)[..., None]

            brdf_color = row_mask_sum(brdf_weight, ray_mask) / ray_count
            tinted_ref_rgb = row_mask_sum(incoming_light * brdf_weight, ray_mask) / ray_count
            spec[bounce_mask] = row_mask_sum(incoming_light, ray_mask) / ray_count
            # ic(incoming_light.mean(), tinted_ref_rgb.mean())

            if self.detach_bg:
                tinted_ref_rgb.detach_()

            reflect_rgb[bounce_mask] = tinted_ref_rgb
            brdf_rgb[bounce_mask] = brdf_color
            # brdf_rgb[bounce_mask] = row_mask_sum((L * eN).sum(dim=-1, keepdim=True).expand(-1, 3), ray_mask) / ray_count

        rgb = reflect_rgb + diffuse
        # ic(rgb.mean(), diffuse.mean())
        debug['diffuse'] = diffuse
        debug['roughness'] = matprop['r1']
        debug['tint'] = brdf_rgb
        debug['spec'] = spec
        return rgb, debug
