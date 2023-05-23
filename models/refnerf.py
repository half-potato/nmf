import numpy as np
import torch
from icecream import ic

from mutils import normalize


class RefNeRF(torch.nn.Module):
    def __init__(self, app_dim, diffuse_module, ref_module, anoise, detach_N_iters):
        super().__init__()
        self.diffuse_module = diffuse_module(in_channels=app_dim)
        self.ref_module = ref_module(in_channels=app_dim)
        self.anoise = anoise
        self.detach_N_iters = detach_N_iters
        self.detach_N = True
        self.outputs = {"diffuse": 3, "roughness": 1, "tint": 3, "spec": 3}
        self.needs_normals = lambda x: True

        self.mean_ratios = None
        self.max_retrace_rays = []

    def get_optparam_groups(self, lr_scale=1):
        grad_vars = []
        grad_vars += [
            {
                "params": self.diffuse_module.parameters(),
                "lr": self.diffuse_module.lr * lr_scale,
            }
        ]
        grad_vars += [
            {
                "params": self.ref_module.parameters(),
                "lr": self.ref_module.lr * lr_scale,
            }
        ]
        return grad_vars

    def calibrate(self, args, xyz, feat, bg_brightness, save_config=True):
        self.diffuse_module.calibrate(
            bg_brightness,
            False,
            xyz,
            normalize(torch.rand_like(xyz[:, :3])),
            feat,
        )
        if save_config:
            args.model.arch.model.diffuse_module.diffuse_bias = (
                self.diffuse_module.diffuse_bias
            )
            args.model.arch.model.diffuse_module.roughness_bias = (
                self.diffuse_module.roughness_bias
            )
        return args

    def check_schedule(self, iter, batch_mul, **kwargs):
        if iter > batch_mul * self.detach_N_iters:
            self.detach_N = False
        return False

    def recover_envmap(self, res, xyz, roughness=None):
        device = xyz.device
        app_feature = self.rf.compute_appfeature(xyz.reshape(1, -1))
        B = 2 * res * res
        staticdir = torch.zeros((B, 3), device=device)
        staticdir[:, 0] = 1
        app_features = app_feature.reshape(1, -1).expand(B, app_feature.shape[-1])
        xyz_samp = xyz.reshape(1, -1).expand(B, xyz.shape[-1])

        ele_grid, azi_grid = torch.meshgrid(
            torch.linspace(-np.pi / 2, np.pi / 2, res, dtype=torch.float32),
            torch.linspace(-np.pi, np.pi, 2 * res, dtype=torch.float32),
            indexing="ij",
        )
        # each col of x ranges from -pi/2 to pi/2
        # each row of y ranges from -pi to pi
        ang_vecs = torch.stack(
            [
                torch.cos(ele_grid) * torch.cos(azi_grid),
                torch.cos(ele_grid) * torch.sin(azi_grid),
                -torch.sin(ele_grid),
            ],
            dim=-1,
        ).to(device)

        if self.ref_module is not None:
            roughness = (
                1
                / np.pi
                * torch.ones(
                    (app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device
                )
            )
            # roughness = matprop['roughness'] if roughness is None else roughness * torch.ones((app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            viewdotnorm = torch.ones(
                (app_features.shape[0], 1), dtype=xyz.dtype, device=xyz.device
            )
            envmap = (
                self.ref_module(
                    xyz_samp,
                    staticdir,
                    app_features,
                    refdirs=ang_vecs.reshape(-1, 3),
                    roughness=roughness,
                    viewdotnorm=viewdotnorm,
                )
            ).reshape(res, 2 * res, 3)
        else:
            envmap = torch.zeros(res, 2 * res, 3)
        # if self.diffuse_module is not None:
        #     color, tint, matprop = self.diffuse_module(xyz_samp, ang_vecs.reshape(-1, 3), app_features)
        #     color = (color).reshape(res, 2*res, 3)/2
        # else:
        #     color = torch.zeros(res, 2*res, 3)

        return self.tonemap(envmap).clamp(0, 1)  # , self.tonemap(color).clamp(0, 1)

    def update_n_samples(self, n_samples):
        return
        # assert(len(self.target_num_samples) == len(self.max_retrace_rays))
        if len(n_samples) == len(self.max_retrace_rays):
            ratios = [
                n_rays / n_sample if n_sample > 0 else None
                for n_rays, n_sample in zip(self.max_retrace_rays, n_samples)
            ]
            if self.mean_ratios is None:
                self.mean_ratios = ratios
            else:
                self.mean_ratios = [
                    (
                        min(0.1 * ratio + 0.9 * mean_ratio, 1, ratio)
                        if ratio is not None
                        else mean_ratio
                    )
                    if mean_ratio is not None
                    else ratio
                    for ratio, mean_ratio in zip(ratios, self.mean_ratios)
                ]
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
        debug = {}
        noise_app_features = app_features + torch.randn_like(app_features) * self.anoise
        diffuse, tint, matprop = self.diffuse_module(xyzs, viewdirs, app_features)

        # calculate reflected ray direction
        VdotN = (-viewdirs * normals).sum(-1, keepdim=True)
        refdirs = 2 * VdotN * normals + viewdirs
        viewdotnorm = (viewdirs * normals).sum(dim=-1, keepdim=True)

        roughness = matprop["r1"].squeeze(-1)
        if self.detach_N:
            refdirs.detach_()
            viewdotnorm.detach_()

        ref_col = self.ref_module(
            xyzs,
            viewdirs,
            noise_app_features,
            refdirs=refdirs,
            roughness=roughness,
            viewdotnorm=viewdotnorm,
        )
        reflect_rgb = tint * ref_col

        debug["diffuse"] = diffuse
        debug["tint"] = tint
        debug["spec"] = ref_col
        debug["roughness"] = matprop["r1"]

        return (reflect_rgb + diffuse).clip(0, 1), debug
