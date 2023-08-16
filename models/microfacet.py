import math

import torch
from icecream import ic

from modules import sh
from modules.pt_selectors import select_bounces
from modules.row_mask_sum import row_mask_sum
from mutils import normalize


class Microfacet(torch.nn.Module):
    def __init__(
        self,
        app_dim,
        diffuse_module,
        brdf,
        brdf_sampler,
        anoise,
        max_brdf_rays,
        target_num_samples,
        russian_roulette,
        percent_bright,
        cold_start_bg_iters,
        detach_N_iters,
        min_rough_start=0,
        min_rough_decay=1,
        start_std=0,
        std_decay=1,
        std_decay_interval=10,
        conserve_energy=True,
        no_emitters=True,
        diffuse_mixing_mode="lambda",
        visibility_module=None,
        max_retrace_rays=[],
        bright_sampler=None,
        freeze=False,
        rays_per_ray=512,
        test_rays_per_ray=512,
    ):
        super().__init__()
        self.diffuse_module = diffuse_module(in_channels=app_dim)
        self.brdf = brdf(in_channels=app_dim)
        self.brdf_sampler = brdf_sampler(max_samples=1024)
        self.bright_sampler = bright_sampler
        self.visibility_module = visibility_module
        self.freeze = freeze

        self.needs_normals = lambda x: True
        self.conserve_energy = conserve_energy
        self.brdf.init_val = 0.5 if self.conserve_energy else 0.25
        self.no_emitters = no_emitters
        self.min_rough = min_rough_start
        self.min_rough_decay = min_rough_decay

        self.std = start_std
        self.std_decay = std_decay
        self.std_decay_interval = std_decay_interval

        self.anoise = anoise
        self.russian_roulette = russian_roulette
        self.target_num_samples = target_num_samples
        self.max_brdf_rays = max_brdf_rays
        self.max_retrace_rays = max_retrace_rays
        self.percent_bright = percent_bright
        self.cold_start_bg_iters = cold_start_bg_iters
        self.conserve_energy = conserve_energy
        self.diffuse_mixing_mode = diffuse_mixing_mode
        self.detach_N_iters = detach_N_iters
        self.detach_N = True
        self.rays_per_ray = rays_per_ray
        self.test_rays_per_ray = test_rays_per_ray
        self.outputs = {"diffuse": 3, "roughness": 1, "tint": 3, "spec": 3}

        self.mean_ratios = None
        self.ratio_list = None

    def calibrate(self, args, xyz, feat, bg_brightness, save_config=True):
        self.diffuse_module.calibrate(
            bg_brightness,
            self.conserve_energy,
            xyz,
            normalize(torch.rand_like(xyz[:, :3])),
            feat,
        )
        self.brdf.calibrate(feat, bg_brightness)
        if save_config:
            args.model.arch.model.brdf.bias = self.brdf.bias
            args.model.arch.model.diffuse_module.diffuse_bias = (
                self.diffuse_module.diffuse_bias
            )
            args.model.arch.model.diffuse_module.roughness_bias = (
                self.diffuse_module.roughness_bias
            )
        return args

    def get_optparam_groups(self, lr_scale=1):
        grad_vars = []
        if not self.freeze:
            grad_vars += [
                {
                    "params": self.diffuse_module.parameters(),
                    "lr": self.diffuse_module.lr * lr_scale,
                }
            ]
            grad_vars += [
                {"params": self.brdf.parameters(), "lr": self.brdf.lr * lr_scale}
            ]
        return grad_vars

    def check_schedule(self, iter, batch_mul, **kwargs):
        # if self.bright_sampler is not None:
        #     self.bright_sampler.check_schedule(iter, batch_mul, bg_module)
        if iter % 10 == 0:
            self.min_rough *= self.min_rough_decay
        if iter > batch_mul * self.detach_N_iters:
            self.detach_N = False
        if iter % self.std_decay_interval == 0:
            self.std *= self.std_decay
        return False

    @torch.no_grad()
    def graph_brdfs(self, xyzs, viewdirs, app_features, res):
        device = app_features.device

        # incoming light directions
        ele_grid, azi_grid = torch.meshgrid(
            torch.linspace(-math.pi / 2, math.pi / 2, res, dtype=torch.float32),
            torch.linspace(0, 2 * math.pi, 2 * res, dtype=torch.float32),
            indexing="ij",
        )
        ang_vecs = (
            torch.stack(
                [
                    -torch.sin(ele_grid),
                    torch.cos(ele_grid) * torch.sin(azi_grid),
                    torch.cos(ele_grid) * torch.cos(azi_grid),
                ],
                dim=-1,
            )
            .reshape(1, 1, -1, 3)
            .to(device)
        )

        assert xyzs.shape[0] == viewdirs.shape[0]
        assert xyzs.shape[0] == app_features.shape[0]

        diffuse, tint, matprop = self.diffuse_module(
            xyzs, viewdirs, app_features, std=0
        )

        n_angs = ang_vecs.shape[-2]
        n_views = viewdirs.shape[0]
        n_features = app_features.shape[0]
        # N = n_features * n_views * n_angs

        L = ang_vecs.expand(n_features, n_views, n_angs, 3)
        eV = viewdirs.reshape(1, -1, 1, 3).expand(n_features, n_views, n_angs, 3)
        halfvec = normalize((L + eV) / 2)
        r1 = (
            matprop["r1"]
            .reshape(-1, 1, 1)
            .expand(n_features, n_views, n_angs)
            .reshape(-1, 1)
        )
        r2 = (
            matprop["r2"]
            .reshape(-1, 1, 1)
            .expand(n_features, n_views, n_angs)
            .reshape(-1, 1)
        )
        efeatures = app_features.reshape(n_features, 1, 1, -1).expand(
            n_features, n_views, n_angs, app_features.shape[-1]
        )
        eN = (
            torch.tensor([0.0, 0.0, 1.0], device=device)
            .reshape(1, 1, 1, 3)
            .expand(n_features, n_views, n_angs, 3)
        )
        proportion = (
            matprop["r1"]
            .reshape(-1, 1, 1)
            .expand(n_features, n_views, n_angs)
            .reshape(-1, 1)
        )

        # normal must be z up so we don't have to create a row basis and rotate everything
        diffvec = L

        brdf_weight = self.brdf(
            eV.reshape(-1, 3),
            L.reshape(-1, 3),
            eN.reshape(-1, 3),
            halfvec.reshape(-1, 3),
            L.reshape(-1, 3),
            halfvec.reshape(-1, 3),
            diffvec.reshape(-1, 3),
            efeatures.reshape(-1, app_features.shape[-1]),
            r1.reshape(-1),
            r2.reshape(-1),
        )

        # next, compute GGX distribution because it's not represented here
        brdf_colors = (
            self.brdf_sampler.compute_prob(
                L.reshape(-1, 3),
                eV.reshape(-1, 3),
                halfvec.reshape(-1, 3),
                r1.reshape(-1, 1),
                r2.reshape(-1, 1),
            ).reshape(-1, 1)
            * brdf_weight
        )

        # add indicator of view direction
        viewdir_ind = (
            (L * eV)
            .sum(dim=-1)
            .reshape(n_features * n_views, n_angs)
            .max(dim=1)
            .indices
        )
        brdf_colors = brdf_colors.reshape(n_features * n_views, n_angs, 3)
        brdf_colors[range(n_features * n_views), viewdir_ind, 0] = 0
        brdf_colors[range(n_features * n_views), viewdir_ind, 1] = 1
        brdf_colors[range(n_features * n_views), viewdir_ind, 2] = 0

        # reshape image
        brdf_colors = brdf_colors.reshape(n_features, n_views, res, 2 * res, 3)
        im = brdf_colors.permute(0, 2, 1, 3, 4).reshape(
            n_features * res, 2 * n_views * res, 3
        )
        return im

    def reset_counter(self):
        self.max_retrace_rays = [1000]
        self.mean_ratios = None
        self.ratio_list = None

    def update_n_samples(self, n_samples):
        # assert(len(self.target_num_samples) == len(self.max_retrace_rays))
        if len(n_samples) == len(self.max_retrace_rays):
            ratios = [
                (n_rays / n_sample) if n_sample > 0 else 1e-3
                for n_rays, n_sample in zip(self.max_retrace_rays, n_samples)
            ]
            if self.ratio_list is None:
                self.ratio_list = [[r, 1e-3] if r is not None else [] for r in ratios]
            else:
                self.ratio_list = [
                    [r for r in ([ratio] + rlist) if r is not None][:20]
                    for ratio, rlist in zip(ratios, self.ratio_list)
                ]
            self.mean_ratios = [
                min(rlist) if len(rlist) > 0 else None for rlist in self.ratio_list
            ]
            # ic(self.mean_ratios, self.ratio_list)
            # ic(n_samples, self.max_retrace_rays)
            self.max_retrace_rays = [
                min(int(target * ratio + 1), maxv) if ratio is not None else prev
                for target, ratio, maxv, prev in zip(
                    self.target_num_samples,
                    self.mean_ratios,
                    self.max_brdf_rays[:-1],
                    self.max_retrace_rays,
                )
            ]
            # ic(ratios, self.mean_ratios, n_samples, self.max_retrace_rays, self.target_num_samples)

    def forward(
        self,
        xyzs,
        xyzs_normed,
        app_features,
        viewdirs,
        normals,
        weights,
        app_mask,
        B,
        render_reflection,
        bg_module,
        is_train,
        recur,
        eps=torch.finfo(torch.float32).eps,
    ):
        # xyzs: (M, 4)
        # viewdirs: (M, 3)
        # normals: (M, 3)
        # weights: (M)
        # B: number of rays being cast
        # recur: recursion counter
        # render_reflection: function that casts out rays to allow for recursion
        debug = {}
        device = xyzs.device

        noise_app_features = app_features + torch.randn_like(app_features) * self.anoise
        std = self.std if is_train else 0
        albedo, tint, matprop = self.diffuse_module(
            xyzs_normed, viewdirs, app_features, std=std
        )

        # compute spherical harmonic coefficients for the background
        if self.no_emitters:
            with torch.no_grad():
                coeffs, conv_coeffs = bg_module.get_spherical_harmonics(100)
                evaled = sh.eval_sh_bases(conv_coeffs.shape[0], normals)
                E = (
                    (
                        conv_coeffs.reshape(1, -1, 3)
                        * evaled.reshape(evaled.shape[0], -1, 1)
                    )
                    .sum(dim=1)
                    .detach()
                )
            diffuse = albedo * E
        else:
            diffuse = albedo

        # pick rays to bounce
        num_brdf_rays = (
            self.max_brdf_rays[recur] if not is_train else self.max_brdf_rays[recur]
        )  # // B
        rays_per_ray = self.rays_per_ray if is_train else self.test_rays_per_ray
        # num_brdf_rays = B * rays_per_ray

        bounce_mask, ray_mask = select_bounces(
            weights,
            app_mask,
            num_brdf_rays,
            self.percent_bright,
            rays_per_ray if recur == 0 else None,
        )
        # ic(ray_mask.sum(), recur, num_brdf_rays)

        reflect_rgb = torch.zeros_like(diffuse)
        brdf_rgb = torch.zeros_like(diffuse)
        spec = torch.zeros_like(diffuse)

        ray_xyz = (
            xyzs[bounce_mask][..., :3]
            .reshape(-1, 1, 3)
            .expand(-1, ray_mask.shape[1], 3)
        )
        if (
            bounce_mask.any()
            and ray_mask.any()
            and ray_xyz.shape[0] == ray_mask.shape[0]
        ):
            ri, rj = torch.where(ray_mask)
            bN = normals[bounce_mask]
            if self.detach_N:
                bN.detach_()
            bV = -viewdirs[bounce_mask]
            # align normals
            bN = bN * (bV * bN).sum(dim=-1, keepdim=True).sign()
            # r1 = matprop['r1'][bounce_mask]*0 + 0.0001
            # r2 = matprop['r2'][bounce_mask]*0 + 0.0001
            r1 = matprop["r1"][bounce_mask]
            r2 = matprop["r1"][bounce_mask]
            if is_train:
                r1 = r1.clip(min=self.min_rough)
                r2 = r2.clip(min=self.min_rough)

            n = ray_xyz.shape[0]
            m = ray_mask.shape[1]
            angs = self.brdf_sampler.draw(n, m).to(device)
            u1 = angs[..., 0]
            u2 = angs[..., 1]
            L, row_world_basis, lpdf = self.brdf_sampler.sample(
                u1, u2, bV, bN, r1, r2, ray_mask
            )

            n = ray_xyz.shape[0]
            m = ray_mask.shape[1]

            eV = bV.reshape(-1, 1, 3).expand(-1, m, 3)[ri, rj]
            eN = bN.reshape(-1, 1, 3).expand(-1, m, 3)[ri, rj]
            ea1 = r1.expand(ray_mask.shape)[ri, rj]
            ea2 = r2.expand(ray_mask.shape)[ri, rj]
            efeatures = (
                noise_app_features[bounce_mask]
                .reshape(n, 1, -1)
                .expand(n, m, -1)[ri, rj]
            )
            exyz = ray_xyz[ri, rj]

            H = normalize((eV + L) / 2)

            # z_up = (
            #     torch.tensor([0.0, 0.0, 1.0], device=device)
            #     .reshape(1, 3)
            #     .expand(H.shape[0], 3)
            # )
            # x_up = (
            #     torch.tensor([-1.0, 0.0, 0.0], device=device)
            #     .reshape(1, 3)
            #     .expand(H.shape[0], 3)
            # )
            # up = torch.where(H[:, 2:3] < 0.999, z_up, x_up)
            # tangent = normalize(torch.linalg.cross(up, H))
            # bitangent = normalize(torch.linalg.cross(H, tangent))
            # B, 3, 3
            # col_world_basis = torch.stack([tangent, bitangent, H], dim=1).reshape(-1, 3, 3).permute(0, 2, 1)

            diffvec = torch.matmul(
                row_world_basis.permute(0, 2, 1), L.unsqueeze(-1)
            ).squeeze(-1)
            # diffvec = torch.stack(
            #     [
            #         tangent[:, 0] * L[:, 0]
            #         + bitangent[:, 0] * L[:, 1]
            #         + H[:, 0] * L[:, 2],
            #         tangent[:, 1] * L[:, 0]
            #         + bitangent[:, 1] * L[:, 1]
            #         + H[:, 1] * L[:, 2],
            #         tangent[:, 2] * L[:, 0]
            #         + bitangent[:, 2] * L[:, 1]
            #         + H[:, 2] * L[:, 2],
            #     ],
            #     dim=-1,
            # )
            local_v = torch.matmul(
                row_world_basis.permute(0, 2, 1), eV.unsqueeze(-1)
            ).squeeze(-1)
            halfvec = torch.matmul(
                row_world_basis.permute(0, 2, 1), H.unsqueeze(-1)
            ).squeeze(-1)
            samp_prob = (lpdf).exp().reshape(-1, 1)

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

            indiv_num_samples = ray_mask.sum(dim=1, keepdim=True).expand(
                ray_mask.shape
            )[ray_mask]
            mipval = -torch.log(indiv_num_samples.clip(min=1)) - lpdf

            bounce_rays = torch.cat(
                [
                    exyz + L * 5e-3,
                    L,
                ],
                dim=-1,
            )
            n = bounce_rays.shape[0]

            # calculate second part of BRDF
            n, m = ray_mask.shape
            brdf_weight = self.brdf(
                eV,
                L.detach(),
                eN.detach(),
                H.detach(),
                local_v.detach(),
                halfvec.detach(),
                diffvec.detach(),
                efeatures,
                ea1.detach(),
                ea2.detach(),
            )
            ray_count = (ray_mask.sum(dim=1) + 1e-8)[..., None]

            if len(self.max_retrace_rays) > recur:
                # decide which rays should get more bounces based on how much they contribute to the final color of the ray
                num_retrace_rays = self.max_retrace_rays[recur]  # // B
                num_retrace_rays = min(brdf_weight.shape[0], num_retrace_rays)
                with torch.no_grad():
                    per_sample_factor = (
                        weights[app_mask][bounce_mask].reshape(-1, 1) / ray_count
                    )
                    per_ray_factor = (
                        brdf_weight.max(dim=-1, keepdim=True).values
                        * ((eV * eN).sum(dim=-1, keepdim=True) > 0)
                        * samp_prob
                    )

                    color_contribution = (
                        per_ray_factor.reshape(-1)
                        * per_sample_factor.expand(ray_mask.shape)[ri, rj]
                    )
                    # normalize color contribution
                    color_contribution = (
                        color_contribution / color_contribution.sum() * num_retrace_rays
                    )
                    if self.visibility_module is not None:
                        ray_xyzs_normed = (
                            xyzs_normed[bounce_mask][..., :3]
                            .reshape(-1, 1, 3)
                            .expand(-1, ray_mask.shape[1], 3)[ri, rj]
                        )
                        color_contribution *= 1 - self.visibility_module(
                            ray_xyzs_normed, L
                        )
                    color_contribution += torch.rand_like(color_contribution)
                    cc_as = color_contribution.argsort()
                    M = max(cc_as.shape[0] - num_retrace_rays, 0)
                    retrace_ray_inds = cc_as[M:]

                    if self.russian_roulette:
                        num_retrace = torch.zeros((ray_mask.shape[0]), device=device)
                        rii = ri[retrace_ray_inds]
                        num_retrace.scatter_add_(
                            0,
                            rii,
                            torch.ones(
                                (1), dtype=num_retrace.dtype, device=device
                            ).expand(rii.shape),
                        )

                        # update ray_count to reflect the new number of rays
                        min_n = 0
                        rtmask = num_retrace > min_n
                        ray_count[rtmask] = num_retrace[rtmask, None]

                        retrace_mask = torch.zeros(
                            color_contribution.shape, device=device, dtype=bool
                        )
                        retrace_mask[retrace_ray_inds] = True
                        lrtmask = rtmask.reshape(-1, 1).expand(ray_mask.shape)[ray_mask]
                        notrace_mask = ~retrace_mask & ~lrtmask

                        (notrace_ray_inds,) = torch.where(notrace_mask)
                    else:
                        notrace_ray_inds = cc_as[:M]

                # ic(
                #     notrace_ray_inds,
                #     retrace_ray_inds,
                #     ray_mask.sum(),
                #     cc_as.shape,
                #     num_retrace_rays,
                # )
                # retrace some of the rays
                incoming_light = torch.zeros((bounce_rays.shape[0], 3), device=device)
                if len(retrace_ray_inds) > 0:
                    incoming_light[retrace_ray_inds], bg_vis = render_reflection(
                        bounce_rays[retrace_ray_inds],
                        mipval[retrace_ray_inds],
                        retrace=True,
                    )
                if len(notrace_ray_inds) > 0:
                    incoming_light[notrace_ray_inds], _ = render_reflection(
                        bounce_rays[notrace_ray_inds],
                        mipval[notrace_ray_inds],
                        retrace=False,
                    )
            else:
                incoming_light, _ = render_reflection(
                    bounce_rays, mipval, retrace=False
                )

            eray_count = (
                ray_count.reshape(-1, 1)
                .expand(ray_mask.shape)[ray_mask]
                .reshape(-1, 1)
                .clip(min=1)
            )
            brdf_color = row_mask_sum(brdf_weight / eray_count, ray_mask)  # / ray_count
            spec[bounce_mask] = row_mask_sum(incoming_light / eray_count, ray_mask)
            brdf_rgb[bounce_mask] = brdf_color

            brdf_rgb[bounce_mask] = brdf_color
            if self.diffuse_mixing_mode == "fresnel_ind":
                R0 = (
                    matprop["f0"][bounce_mask]
                    .reshape(-1, 1, 3)
                    .expand(-1, m, 3)[ri, rj]
                )
                # itint = tint[bounce_mask].reshape(-1, 1, 3).expand(-1, m, 3)[ri, rj]
                ediffuse = (
                    diffuse[bounce_mask].reshape(-1, 1, 3).expand(-1, m, 3)[ri, rj]
                )
                costheta = (-eV * H).sum(dim=-1, keepdim=True).abs()
                spec_reflectance = (
                    R0 + (1 - R0) * (1 - costheta).clip(min=0, max=1) ** 5
                )
                comb_rgb = (
                    spec_reflectance * incoming_light
                    + (1 - spec_reflectance) * ediffuse
                )
                reflect_rgb[bounce_mask] = row_mask_sum(comb_rgb / eray_count, ray_mask)
            elif self.diffuse_mixing_mode == "fresnel":
                R0 = (
                    matprop["f0"][bounce_mask]
                    .reshape(-1, 1, 3)
                    .expand(-1, m, 3)[ri, rj]
                )
                # itint = tint[bounce_mask].reshape(-1, 1, 3).expand(-1, m, 3)[ri, rj]
                ediffuse = (
                    diffuse[bounce_mask].reshape(-1, 1, 3).expand(-1, m, 3)[ri, rj]
                )
                costheta = (-eV * H).sum(dim=-1, keepdim=True).abs()
                spec_reflectance = (
                    R0 + (1 - R0) * (1 - costheta).clip(min=0, max=1) ** 5
                )
                comb_rgb = (
                    spec_reflectance * incoming_light * brdf_weight
                    + (1 - spec_reflectance) * ediffuse
                )
                reflect_rgb[bounce_mask] = row_mask_sum(comb_rgb / eray_count, ray_mask)
            else:
                tinted_ref_rgb = row_mask_sum(
                    incoming_light / eray_count * brdf_weight,
                    ray_mask,
                )
                reflect_rgb[bounce_mask] = tinted_ref_rgb

            # ic(
            #     incoming_light.shape,
            #     incoming_light.mean(),
            #     incoming_light.max(),
            #     spec[bounce_mask].min(),
            #     spec[bounce_mask].mean(),
            #     brdf_weight.max(),
            #     brdf_weight.mean(),
            #     tinted_ref_rgb.mean(),
            #     diffuse.min(),
            #     diffuse.max(),
            #     bg_module.mean_color(),
            # )

        if self.diffuse_mixing_mode == "no_diffuse":
            rgb = reflect_rgb
            debug["diffuse"] = diffuse
            debug["tint"] = brdf_rgb
        elif self.diffuse_mixing_mode == "fresnel":
            R0 = matprop["f0"]  # .mean(dim=-1, keepdim=True)
            # R0 = R0 * 0 + 0.04
            costheta = (-viewdirs * normals).sum(dim=-1, keepdim=True).abs()
            spec_reflectance = R0 + (1 - R0) * (1 - costheta).clip(min=0, max=1) ** 5
            # rgb = spec_reflectance * reflect_rgb + (1 - spec_reflectance) * diffuse
            rgb = reflect_rgb
            debug["diffuse"] = (1 - spec_reflectance) * diffuse
            debug["tint"] = spec_reflectance * brdf_rgb
        elif self.diffuse_mixing_mode == "fresnel_ind":
            R0 = matprop["f0"]  # .mean(dim=-1, keepdim=True)
            # R0 = R0 * 0 + 0.04
            costheta = (-viewdirs * normals).sum(dim=-1, keepdim=True).abs()
            spec_reflectance = R0 + (1 - R0) * (1 - costheta).clip(min=0, max=1) ** 5
            # rgb = spec_reflectance * reflect_rgb + (1 - spec_reflectance) * diffuse
            rgb = reflect_rgb
            debug["diffuse"] = (1 - spec_reflectance) * diffuse
            debug["tint"] = spec_reflectance
        elif self.diffuse_mixing_mode == "lambda":
            lam = tint.mean(dim=-1, keepdim=True)
            rgb = lam * reflect_rgb + (1 - lam) * diffuse
            # uh..... to match fresnel_ind behavior. It works stop asking questions
            rgb[~bounce_mask] = 0
            # ic(
            #     rgb[bounce_mask].mean(),
            #     diffuse.mean(),
            #     reflect_rgb[bounce_mask].mean(),
            #     lam.mean(),
            # )
            debug["diffuse"] = diffuse * (1 - lam)
            debug["tint"] = brdf_rgb * lam
        debug["roughness"] = matprop["r1"]
        debug["spec"] = spec
        return rgb, debug
