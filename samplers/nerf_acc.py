import torch
import math
from icecream import ic

from nerfacc import ContractionType, OccupancyGrid, ray_marching

class NerfAccSampler(torch.nn.Module):
    def __init__(self,
                 aabb,
                 near_far,
                 grid_size=128,
                 render_n_samples=1024,
                 update_freq=16,
                 alpha_thres = 0,
                 ):
        super().__init__()
        contraction_type = ContractionType.AABB
        self.aabb = aabb.reshape(-1)
        self.near_far = near_far
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=self.aabb,
            resolution=grid_size,
            contraction_type=contraction_type,
        )
        self.alpha_thres = alpha_thres
        self.update_freq = update_freq
        self.cone_angle = 0.0
        self.stepsize = (
            (self.aabb[3:] - self.aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()

    def check_schedule(self, iteration, batch_mul, rf):
        def occ_eval_fn(x):
            # x is in aabb, unnormalized
            density = rf.compute_densityfeature(x).reshape(-1)
            return density * self.stepsize
        self.occupancy_grid.every_n_step(step=iteration+1, occ_eval_fn=occ_eval_fn)

        self.nSamples = int(rf.nSamples)
        near, far = self.near_far
        self.stepsize = (far - near) / self.nSamples
        return False

    def update(self, rf, init=True):
        def occ_eval_fn(x):
            # x is in aabb, unnormalized
            density = rf.compute_densityfeature(x).reshape(-1)
            return density * self.stepsize
        self.occupancy_grid.every_n_step(step=0, occ_eval_fn=occ_eval_fn)

    def sample(self, rays_chunk, focal, rf, override_near=None, is_train=False,
               dynamic_batch_size=True, stepmul=1, **args):
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
            x = origins[ray_indices] + (t_ends+t_starts)/2 * viewdirs[ray_indices]
            return rf.compute_densityfeature(x).reshape(-1, 1)
        origins = rays_chunk[:, 0:3]
        viewdirs = rays_chunk[:, 3:6]
        ray_indices, t_starts, t_ends = ray_marching(
            origins,
            viewdirs,
            scene_aabb=self.aabb,
            grid=self.occupancy_grid,
            sigma_fn=sigma_fn,
            # near_plane=self.near_far[0] if override_near is None else override_near,
            # far_plane=self.near_far[1],
            near_plane=None,
            far_plane=None,
            render_step_size=self.stepsize/stepmul,
            stratified=is_train,
            cone_angle=self.cone_angle,
            alpha_thre=self.alpha_thres,
        )
        if len(ray_indices) == 0:
            return (torch.empty((0, 4), device=device),
                   torch.zeros((N, 1), dtype=bool, device=device),
                   1,
                   torch.empty((N, 1), device=device),
                   torch.empty((N, 1), device=device),
                   torch.ones((N), dtype=bool, device=device))

        pz_vals = (t_starts + t_ends) / 2
        pdists = t_ends - t_starts
        xyz_sampled = origins[ray_indices] + pz_vals * viewdirs[ray_indices]
        whole_valid = torch.ones((N), dtype=bool, device=device)

        # turn ray_indices into ray_valid
        ele, counts = torch.unique(ray_indices, return_counts=True)
        num_samps = torch.zeros((N), dtype=counts.dtype, device=device)
        num_samps[ele] = counts
        M = counts.max(dim=0).values
        ray_valid = torch.arange(M, device=device).reshape(1, -1) < num_samps.reshape(-1, 1)

        # pad out dists and z_vals
        dists = torch.zeros(ray_valid.shape, device=device)
        z_vals = torch.zeros(ray_valid.shape, device=device)
        dists[ray_valid] = pdists.reshape(-1)
        z_vals[ray_valid] = pz_vals.reshape(-1)

        xyz_sampled_sized = torch.cat([xyz_sampled, torch.zeros((xyz_sampled.shape[0], 1), device=device)], dim=-1)

        return xyz_sampled_sized, ray_valid, M, z_vals, dists, whole_valid