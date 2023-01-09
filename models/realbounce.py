import torch
from modules.pt_selectors import select_bounces
from mutils import normalize
from modules.row_mask_sum import row_mask_sum
from icecream import ic

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

            L, row_world_basis = self.brdf_sampler.sample(
                    bV, bN,
                    r1**2, r2**2, ray_mask)

            # Sample bright spots
            if self.bright_sampler is not None and self.bright_sampler.is_initialized():
                bL, bright_mask = self.bright_sampler.sample(bV, bN, ray_mask, bright_mask)
                pbright_mask = bright_mask[ray_mask]
                L[pbright_mask] = bL[bright_mask]

            n = ray_xyz.shape[0]
            m = ray_mask.shape[1]

            eV = bV.reshape(-1, 1, 3).expand(-1, m, 3)[ray_mask]
            eN = bN.reshape(-1, 1, 3).expand(-1, m, 3)[ray_mask]
            ea = r1.expand(ray_mask.shape)[ray_mask]
            efeatures = noise_app_features[bounce_mask].reshape(n, 1, -1).expand(n, m, -1)[ray_mask]
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
            mipval = self.brdf_sampler.calculate_mipval(H.detach(), eV, eN.detach(), ray_mask, ea**2)

            bounce_rays = torch.cat([
                exyz,
                L,
            ], dim=-1)
            n = bounce_rays.shape[0]
            D = bounce_rays.shape[-1]
            incoming_light = ray_cast_fn(bounce_rays, mipval)

            # calculate second part of BRDF
            n, m = ray_mask.shape
            diffvec = torch.matmul(row_world_basis.permute(0, 2, 1), L.unsqueeze(-1)).squeeze(-1)
            halfvec = torch.matmul(row_world_basis.permute(0, 2, 1), H.unsqueeze(-1)).squeeze(-1)
            # brdf_weight = self.brdf(eV, L, eN, halfvec, diffvec, efeatures, eroughness)
            brdf_weight = self.brdf(eV, L.detach(), eN.detach(), halfvec.detach(), diffvec.detach(), efeatures, ea.detach())
            ray_count = (ray_mask.sum(dim=1)+1e-8)[..., None]

            brdf_color = row_mask_sum(brdf_weight, ray_mask) / ray_count
            tinted_ref_rgb = row_mask_sum(incoming_light * brdf_weight, ray_mask) / ray_count
            spec[bounce_mask] = row_mask_sum(incoming_light, ray_mask) / ray_count

            if self.detach_bg:
                tinted_ref_rgb.detach_()

            reflect_rgb[bounce_mask] = tinted_ref_rgb
            brdf_rgb[bounce_mask] = brdf_color
            # brdf_rgb[bounce_mask] = row_mask_sum((L * eN).sum(dim=-1, keepdim=True).expand(-1, 3), ray_mask) / ray_count

        rgb = reflect_rgb + diffuse
        debug['diffuse'] = diffuse
        debug['roughness'] = matprop['r1']
        debug['tint'] = brdf_rgb
        debug['spec'] = spec
        return rgb, debug
