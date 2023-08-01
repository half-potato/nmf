import math

import torch
from icecream import ic
from nerfacc import OccGridEstimator


class NerfAccSampler(torch.nn.Module):
    def __init__(
        self,
        aabb,
        near_far,
        grid_size=128,
        render_n_samples=1024,
        max_samples=-1,
        multiplier=1,
        test_multiplier=1,
        update_freq=16,
        shrink_iters=[],
        alpha_thres=0,
        ema_decay=0.95,
        levels=1,
        occ_thre=0.01,
        warmup_iters=256,
    ):
        super().__init__()
        self.aabb = aabb.reshape(-1)
        self.near_far = near_far
        self.grid_size = grid_size
        self.levels = levels
        self.occupancy_grid = OccGridEstimator(
            roi_aabb=self.aabb,
            resolution=grid_size,
            levels=levels,
        )
        self.multiplier = multiplier
        self.alpha_thres = alpha_thres
        self.update_freq = update_freq
        self.cone_angle = 0.0
        self.max_samples = max_samples
        self.test_multiplier = test_multiplier
        self.shrink_iters = shrink_iters
        self.warmup_iters = warmup_iters

        self.distance_scale = 25
        self.stepsize = (
            (self.aabb[3:] - self.aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
            / self.multiplier
        ).item()

        self.occ_thre = occ_thre
        self.ema_decay = ema_decay

    def check_schedule(self, iteration, batch_mul, rf):
        def occ_eval_fn(x):
            # x is in aabb, unnormalized
            density = rf.compute_densityfeature(x).reshape(-1)
            return density * self.stepsize * rf.distance_scale

        self.occupancy_grid.update_every_n_steps(
            step=iteration + 1,
            occ_eval_fn=occ_eval_fn,
            ema_decay=self.ema_decay,
            occ_thre=self.occ_thre,
            warmup_steps=self.warmup_iters,
        )

        # if iteration in [i * batch_mul for i in self.shrink_iters]:
        #     x = (
        #         self.occupancy_grid.grid_coords[self.occupancy_grid._binary.reshape(-1)]
        #         / self.occupancy_grid.resolution
        #     )
        #     x = contract_inv(
        #         x,
        #         roi=self.occupancy_grid._roi_aabb,
        #         type=self.occupancy_grid._contraction_type,
        #     )
        #     new_aabb = torch.stack([x.min(dim=0).values, x.max(dim=0).values], dim=0)
        #     rf.shrink(new_aabb, None)
        #     self.update(rf, init=True)
        #     return True

        self.distance_scale = rf.distance_scale
        self.nSamples = int(rf.nSamples * self.multiplier)
        near, far = self.near_far
        self.stepsize = (far - near) / self.nSamples
        return False

    @torch.no_grad()
    def update(self, rf, init=True):
        if init:
            self.occupancy_grid = OccGridEstimator(
                roi_aabb=self.aabb,
                resolution=self.grid_size,
                levels=self.levels,
            ).to(rf.get_device())

        def occ_eval_fn(x):
            # x is in aabb, unnormalized
            density = rf.compute_densityfeature(x).reshape(-1)
            return density * self.stepsize * rf.distance_scale

        self.distance_scale = rf.distance_scale

        self.occupancy_grid.update_every_n_steps(
            step=0,
            occ_eval_fn=occ_eval_fn,
            ema_decay=self.ema_decay,
            occ_thre=self.occ_thre,
        )

    @torch.no_grad()
    def sample(
        self,
        rays_chunk,
        focal,
        rf,
        override_near=None,
        is_train=False,
        dynamic_batch_size=True,
        override_alpha_thres=None,
        stepmul=1,
        **args
    ):
        """
        Parameters:
            rays_chunk: (B, 6) float. (ox, oy, oz, dx, dy, dz) ray origin and direction
            focal: focal length of camera projecting ray. Unused
            ndc_ray: whether to sample ray in normalized device coordinates
            override_near: an override on what the closest point sampled can be
            is_train: varies sampler based sampler_mode
            N_samples: override how many points are sampled
        Returns:
            xyz_sampled: (M, 4) float. premasked valid sample points
            ray_valid: (b, N) bool. mask of which samples are valid
            max_samps = N
            z_vals: (b, N) float. distance along ray to sample
            dists: (b, N) float. distance between samples
            whole_valid: mask into origin rays_chunk of which b rays where able to be fully sampled.
        """
        device = rays_chunk.device
        N = rays_chunk.shape[0]

        def sigma_fn(t_starts, t_ends, ray_indices):
            x = (
                origins[ray_indices]
                + (t_ends + t_starts).reshape(-1, 1) / 2 * viewdirs[ray_indices]
            )
            max_samps = 128**3
            occs = []
            for bx in x.split(max_samps, dim=0):
                occ = rf.compute_densityfeature(bx).reshape(-1, 1)
                occs.append(occ)
            occs = torch.cat(occs, dim=0).reshape(-1) * self.distance_scale
            return occs

        stepmul *= self.test_multiplier if not is_train else 1
        origins = rays_chunk[:, 0:3]
        viewdirs = rays_chunk[:, 3:6]
        ray_indices, t_starts, t_ends = self.occupancy_grid.sampling(
            origins,
            viewdirs,
            sigma_fn=sigma_fn,
            near_plane=self.near_far[0] if override_near is None else override_near,
            far_plane=self.near_far[1],
            # near_plane=None,
            # far_plane=None,
            render_step_size=self.stepsize / stepmul,
            stratified=False,
            cone_angle=self.cone_angle,
            alpha_thre=self.alpha_thres
            if override_alpha_thres is None
            else override_alpha_thres,
        )
        if len(ray_indices) == 0:
            return (
                torch.empty((0, 4), device=device),
                torch.zeros((N, 1), dtype=bool, device=device),
                1,
                torch.empty((N, 1), device=device),
                torch.empty((N, 1), device=device),
                torch.ones((N), dtype=bool, device=device),
            )

        pz_vals = (t_starts + t_ends).reshape(-1, 1) / 2
        pdists = t_ends - t_starts
        xyz_sampled = origins[ray_indices] + pz_vals * viewdirs[ray_indices]

        # turn ray_indices into ray_valid
        ele, counts = torch.unique(ray_indices, return_counts=True)
        num_samps = torch.zeros((N), dtype=counts.dtype, device=device)
        num_samps[ele] = counts
        M = counts.max(dim=0).values
        ray_valid = torch.arange(M, device=device).reshape(1, -1) < num_samps.reshape(
            -1, 1
        )

        # add size
        xyz_sampled = torch.cat(
            [xyz_sampled, torch.zeros((xyz_sampled.shape[0], 1), device=device)], dim=-1
        )

        # pad out dists and z_vals
        dists = torch.zeros(ray_valid.shape, device=device)
        z_vals = torch.zeros(ray_valid.shape, device=device)
        dists[ray_valid] = pdists.reshape(-1)
        z_vals[ray_valid] = pz_vals.reshape(-1)

        if (
            self.max_samples > 0
            and is_train
            and dynamic_batch_size
            and xyz_sampled.shape[0] > self.max_samples
        ):
            xyz_sampled_w = torch.zeros((*ray_valid.shape, 4), device=device)
            xyz_sampled_w[ray_valid] = xyz_sampled
            whole_valid = torch.cumsum(ray_valid.sum(dim=1), dim=0) < self.max_samples
            ray_valid = ray_valid[whole_valid, :]
            xyz_sampled_w = xyz_sampled_w[whole_valid, :]
            z_vals = z_vals[whole_valid, :]
            dists = dists[whole_valid, :]
            xyzs = xyz_sampled_w[ray_valid]
        else:
            whole_valid = torch.ones((N), dtype=bool, device=device)
            xyzs = xyz_sampled
        # ic(xyz_sampled.shape, whole_valid.sum(), N, ray_valid.sum(), is_train, dynamic_batch_size)
        # ic(xyz_sampled.shape)

        return xyzs, ray_valid, M, z_vals, dists, whole_valid
