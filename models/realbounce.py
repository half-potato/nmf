import torch
from modules.pt_selectors import select_bounces
from mutils import normalize
from modules.row_mask_sum import row_mask_sum
from icecream import ic
import math
from modules import sh

class RealBounce(torch.nn.Module):
    def __init__(self, app_dim, diffuse_module, brdf, brdf_sampler, 
                 anoise, max_brdf_rays, target_num_samples, russian_roulette,
                 percent_bright, cold_start_bg_iters, detach_N_iters, visibility_module=None, max_retrace_rays=[], bright_sampler=None):
        super().__init__()
        self.diffuse_module = diffuse_module(in_channels=app_dim)
        self.brdf = brdf(in_channels=app_dim)
        self.brdf_sampler = brdf_sampler(max_samples=1024)
        self.bright_sampler = bright_sampler
        self.visibility_module = visibility_module

        self.anoise = anoise
        self.russian_roulette = russian_roulette
        self.target_num_samples = target_num_samples
        self.max_brdf_rays = max_brdf_rays
        self.max_retrace_rays = max_retrace_rays
        self.percent_bright = percent_bright
        self.cold_start_bg_iters = cold_start_bg_iters
        self.detach_bg = True
        self.detach_N_iters = detach_N_iters
        self.detach_N = True
        self.outputs = {'diffuse': 3, 'roughness': 1, 'tint': 3, 'spec': 3}

        self.mean_ratios = None

    def get_optparam_groups(self, lr_scale=1):
        grad_vars = []
        grad_vars += [{'params': self.diffuse_module.parameters(),
                       'lr': self.diffuse_module.lr*lr_scale}]
        grad_vars += [{'params': self.brdf.parameters(),
                       'lr': self.brdf.lr*lr_scale}]
        return grad_vars

    def check_schedule(self, iter, batch_mul, bg_module, **kwargs):
        # if self.bright_sampler is not None:
        #     self.bright_sampler.check_schedule(iter, batch_mul, bg_module)
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
        brdf_colors = (self.brdf_sampler.compute_prob(halfvec.reshape(-1, 3), eN.reshape(-1, 3), r1.reshape(-1, 1), r2.reshape(-1, 1)).reshape(-1, 1) * brdf_weight)

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

    def update_n_samples(self, n_samples):
        assert(len(self.target_num_samples) == len(self.max_retrace_rays))
        if len(n_samples) == len(self.max_retrace_rays):
            ratios = [n_rays / n_sample if n_sample > 0 else None for n_rays, n_sample in zip(self.max_retrace_rays, n_samples)]
            if self.mean_ratios is None:
                self.mean_ratios = ratios
            else:
                self.mean_ratios = [
                        (min(0.1*ratio + 0.9*mean_ratio, 1, ratio) if ratio is not None else mean_ratio) if mean_ratio is not None else ratio
                        for ratio, mean_ratio in zip(ratios, self.mean_ratios)]
            self.max_retrace_rays = [
                    min(int(target * ratio + 1), maxv) if ratio is not None else prev
                    for target, ratio, maxv, prev in zip(self.target_num_samples, self.mean_ratios, self.max_brdf_rays[:-1], self.max_retrace_rays)]
            # ic(ratios, self.mean_ratios, n_samples, self.max_retrace_rays, self.target_num_samples)


    def forward(self, xyzs, xyzs_normed, app_features, viewdirs, normals, weights, app_mask, B, recur, render_reflection, bg_module, is_train, eps=torch.finfo(torch.float32).eps):
        # xyzs: (M, 4)
        # viewdirs: (M, 3)
        # normals: (M, 3)
        # weights: (M)
        # B: number of rays being cast
        # recur: recursion counter
        # render_reflection: function that casts out rays to allow for recursion
        debug = {}
        device = xyzs.device

        noise_app_features = (app_features + torch.randn_like(app_features) * self.anoise)
        diffuse, tint, matprop = self.diffuse_module(
            xyzs_normed, viewdirs, app_features)

        # compute spherical harmonic coefficients for the background
        with torch.no_grad():
            coeffs, conv_coeffs = bg_module.get_spherical_harmonics(100)
        evaled = sh.eval_sh_bases(coeffs.shape[0], normals)
        E = (conv_coeffs.reshape(1, -1, 3) * evaled.reshape(evaled.shape[0], -1, 1)).sum(dim=1).detach()
        diffuse = diffuse * E

        # pick rays to bounce
        num_brdf_rays = self.max_brdf_rays[recur]# // B

        bounce_mask, ray_mask, bright_mask = select_bounces(
                weights, app_mask, num_brdf_rays, self.percent_bright)

        reflect_rgb = torch.zeros_like(diffuse)
        brdf_rgb = torch.zeros_like(diffuse)
        spec = torch.zeros_like(diffuse)

        ray_xyz = xyzs[bounce_mask][..., :3].reshape(-1, 1, 3).expand(-1, ray_mask.shape[1], 3)
        # ic(ray_mask.shape, ray_mask.sum(), diffuse.shape)
        if bounce_mask.any() and ray_mask.any() and ray_xyz.shape[0] == ray_mask.shape[0]:
            ri, rj = torch.where(ray_mask)
            bN = normals[bounce_mask]
            if self.detach_N:
                bN.detach_()
            bV = -viewdirs[bounce_mask]
            # r1 = matprop['r1'][bounce_mask]*0 + 0.0001
            # r2 = matprop['r2'][bounce_mask]*0 + 0.0001
            r1 = matprop['r1'][bounce_mask]
            r2 = matprop['r2'][bounce_mask]
            proportion = matprop['proportion'][bounce_mask]

            L, row_world_basis, lpdf = self.brdf_sampler.sample(
                    bV, bN,
                    r1, r2, ray_mask, proportion=proportion)

            n = ray_xyz.shape[0]
            m = ray_mask.shape[1]

            eV = bV.reshape(-1, 1, 3).expand(-1, m, 3)[ri, rj]
            eN = bN.reshape(-1, 1, 3).expand(-1, m, 3)[ri, rj]
            ea1 = r1.expand(ray_mask.shape)[ri, rj]
            ea2 = r2.expand(ray_mask.shape)[ri, rj]
            efeatures = noise_app_features[bounce_mask].reshape(n, 1, -1).expand(n, m, -1)[ri, rj]
            eproportion = proportion.expand(ray_mask.shape)[ri, rj]
            exyz = ray_xyz[ri, rj]

            importance_samp_correction = torch.ones((L.shape[0], 1), device=device)
            # Sample bright spots
            # ic(ray_mask.sum(), bright_mask.sum(), num_brdf_rays)
            if self.bright_sampler is not None:
                bL, bsamp_prob, brightsum = self.bright_sampler.sample(bg_module, bright_mask.sum())
                bsamp_prob = bsamp_prob * self.percent_bright
                pbright_mask = bright_mask[ri, rj]
                L[pbright_mask] = bL


            H = normalize((eV+L)/2)
            diffvec = torch.matmul(row_world_basis.permute(0, 2, 1), L.unsqueeze(-1)).squeeze(-1)
            halfvec = torch.matmul(row_world_basis.permute(0, 2, 1), H.unsqueeze(-1)).squeeze(-1)
            # samp_prob = self.brdf_sampler.compute_prob(halfvec, eN, ea1.reshape(-1, 1), ea2.reshape(-1, 1), proportion=proportion)
            samp_prob = (lpdf).exp().reshape(-1, 1)

            if self.bright_sampler is not None:
                samp_prob = samp_prob * (1-self.percent_bright)
                p1 = bsamp_prob.reshape(-1, 1)
                p2 = samp_prob[pbright_mask].reshape(-1, 1)
                # the first step is to convert everything to probabilities in the area measure (area on the surface of the sphere)
                # bsamp_prob is in the area measure
                # samp_prob is in the solid angle measure
                # turns out, envmaps are in solid angle measure as well, so no need

                weight = p1*p1 / (p1*p1+p2*p2).clip(min=eps)
                importance_samp_correction[pbright_mask] = p2 / p1.clip(min=eps) * weight
                # ic(importance_samp_correction[pbright_mask])

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

            indiv_num_samples = ray_mask.sum(dim=1, keepdim=True).expand(ray_mask.shape)[ray_mask]
            mipval = -torch.log(indiv_num_samples.clip(min=1)) - lpdf
            # mipval = self.brdf_sampler.calculate_mipval(H.detach(), eV, eN.detach(), ray_mask, ea1, ea2, proportion=eproportion)

            bounce_rays = torch.cat([
                exyz + L*5e-3,
                L,
            ], dim=-1)
            n = bounce_rays.shape[0]
            D = bounce_rays.shape[-1]

            # calculate second part of BRDF
            n, m = ray_mask.shape
            # brdf_weight = self.brdf(eV, L, eN, halfvec, diffvec, efeatures, ea)
            brdf_weight = self.brdf(eV, L.detach(), eN.detach(), halfvec.detach(), diffvec.detach(), efeatures, ea1.detach())
            ray_count = (ray_mask.sum(dim=1)+1e-8)[..., None]

            if len(self.max_retrace_rays) > recur:
                # decide which rays should get more bounces based on how much they contribute to the final color of the ray
                num_retrace_rays = self.max_retrace_rays[recur]# // B
                num_retrace_rays = min(brdf_weight.shape[0], num_retrace_rays)
                with torch.no_grad():
                    per_sample_factor = weights[app_mask][bounce_mask].reshape(-1, 1) / ray_count
                    per_ray_factor = brdf_weight.max(dim=-1, keepdim=True).values * ((eV * eN).sum(dim=-1, keepdim=True) > 0) * samp_prob

                    color_contribution = per_ray_factor.reshape(-1) * per_sample_factor.expand(ray_mask.shape)[ri, rj]
                    if self.visibility_module is not None:
                        ray_xyzs_normed = xyzs_normed[bounce_mask][..., :3].reshape(-1, 1, 3).expand(-1, ray_mask.shape[1], 3)[ri, rj]
                        color_contribution *= 1-self.visibility_module(ray_xyzs_normed, L)
                    color_contribution += 0.2*torch.rand_like(color_contribution)
                    cc_as = color_contribution.argsort()
                    retrace_ray_inds = cc_as[cc_as.shape[0]-num_retrace_rays:]

                    if self.russian_roulette:
                        num_retrace = torch.zeros((ray_mask.shape[0]), device=device)
                        rii = ri[retrace_ray_inds]
                        num_retrace.scatter_add_(0, rii, torch.ones((1), dtype=num_retrace.dtype, device=device).expand(rii.shape))


                        # update ray_count to reflect the new number of rays
                        min_n = 0
                        rtmask = num_retrace > min_n
                        ray_count[rtmask] = num_retrace[rtmask, None]

                        retrace_mask = torch.zeros(color_contribution.shape, device=device, dtype=bool)
                        retrace_mask[retrace_ray_inds] = True
                        lrtmask = rtmask.reshape(-1, 1).expand(ray_mask.shape)[ray_mask]
                        notrace_mask = ~retrace_mask & ~lrtmask

                        notrace_ray_inds, = torch.where(notrace_mask)
                    else:
                        notrace_ray_inds = cc_as[:-num_retrace_rays]

                # retrace some of the rays
                incoming_light = torch.zeros((bounce_rays.shape[0], 3), device=device)
                if len(retrace_ray_inds) > 0:
                    incoming_light[retrace_ray_inds], bg_vis = render_reflection(bounce_rays[retrace_ray_inds], mipval[retrace_ray_inds], retrace=True)
                    # bg vis is high when bg is visible
                    if self.visibility_module is not None:
                        self.visibility_module.fit(ray_xyzs_normed[retrace_ray_inds], L[retrace_ray_inds], bg_vis > 0.9)
                    # ic((bg_vis).float().mean())
                if len(notrace_ray_inds) > 0:
                    incoming_light[notrace_ray_inds], _ = render_reflection(bounce_rays[notrace_ray_inds], mipval[notrace_ray_inds], retrace=False)
            else:
                incoming_light, _ = render_reflection(bounce_rays, mipval, retrace=False)

            # ic(incoming_light[pbright_mask].mean(), incoming_light[~pbright_mask].mean())

            if self.bright_sampler is not None:
                p1 = incoming_light.mean(dim=-1, keepdim=True) / (2 * math.pi * math.pi) / brightsum
                p2 = samp_prob
                weight = p2*p2 / (p1*p1+p2*p2).clip(min=eps)
                importance_samp_correction[~pbright_mask] = weight[~pbright_mask]


            brdf_color = row_mask_sum(brdf_weight, ray_mask) / ray_count
            tinted_ref_rgb = row_mask_sum(incoming_light * brdf_weight * importance_samp_correction, ray_mask) / ray_count
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
