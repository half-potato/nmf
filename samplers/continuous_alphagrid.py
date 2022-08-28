import torch
import math
import raymarching_full as raymarching 
import torch.nn.functional as F
from numba import jit
import numpy as np
from icecream import ic
import time

class ContinuousAlphagrid(torch.nn.Module):
    def __init__(self,
                 bound=2.0,
                 near_far=[0.2, 6],
                 threshold=0.002,
                 multiplier=1, 
                 sample_mode='multi_jitter',
                 test_sample_mode=None,
                 update_freq=16,
                 max_samples=int(1.1e6),
                 grid_size=128):
        super().__init__()

        self.bound = bound
        self.update_freq = update_freq
        self.cascade = int(1 + math.ceil(math.log2(bound)))
        self.grid_size = grid_size
        self.multiplier = int(multiplier)
        # self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = grid_size
        self.near_far = near_far
        self.threshold = threshold
        self.active_density_thresh = threshold
        self.max_samples = max_samples 
        self.stepsize = 0.003383

        self.sample_mode = sample_mode
        self.test_sample_mode = sample_mode if test_sample_mode is None else test_sample_mode
        # self.stepsize = 0.005
        # aabb_train = torch.FloatTensor(aabb)
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        # density grid
        density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
        self.register_buffer('step_counter', step_counter)
        self.mean_count = 0
        self.local_step = 0

    def sample_ray_ndc(self, rays_o, rays_d, focal, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            l = torch.rand_like(interpx)
            interpx += l.to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (
            rays_pts > self.aabb[1])).any(dim=-1)

        # add size
        rays_pts = torch.cat([rays_pts, interpx.unsqueeze(-1)/focal], dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, focal, is_train=True, override_near=None, N_samples=-1, N_env_samples=-1):
        # focal: ratio of meters to pixels at a distance of 1 meter
        N_samples = N_samples if N_samples > 0 else self.nSamples
        # N_env_samples = N_env_samples if N_env_samples > 0 else self.nEnvSamples
        N_env_samples = 0
        device = rays_o.device
        stepsize = self.stepSize
        near, far = self.near_far
        if override_near is not None:
            near = override_near
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1].to(rays_o) - rays_o) / vec
        rate_b = (self.aabb[0].to(rays_o) - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        # t_min = near * torch.ones_like(t_min)

        rng = torch.arange(N_samples, device=rays_o.device)[None].float()
        # extend rng to sample towards infinity
        if N_env_samples > 0:
            ext_rng = N_samples + N_env_samples / \
                torch.linspace(1, 1/N_env_samples, N_env_samples,
                               device=rays_o.device)[None].float()
            rng = torch.cat([rng, ext_rng], dim=1)

        sample_mode = self.sample_mode if is_train else self.test_sample_mode                                                                                                                                      
        match sample_mode:                                                                                                                                                                                         
            case 'multi_jitter':                                                                                                                                                                                   
                rng = rng.repeat(rays_d.shape[-2], 1)                                                                                                                                                              
                brng = rng.reshape(-1, N_samples+N_env_samples)                                                                                                                                                    
                r = torch.rand_like(brng)                                                                                                                                                                          
                brng = brng + r                                                                                                                                                                                    
                rng = brng.reshape(-1, N_samples+N_env_samples)                                                                                                                                                    
                step = stepsize * rng                                                                                                                                                                              

            case 'single_jitter':                                                                                                                                                                                  
                rng = rng.repeat(rays_d.shape[-2], 1)                                                                                                                                                              
                brng = rng.reshape(-1, N_samples+N_env_samples)                                                                                                                                                    
                r = torch.rand_like(brng[:, 0:1])                                                                                                                                                                  
                r = torch.rand_like(brng)                                                                                                                                                                          
                brng = brng + r                                                                                                                                                                                    
                rng = brng.reshape(-1, N_samples+N_env_samples)                                                                                                                                                    
                step = stepsize * rng                                                                                                                                                                              

            case 'cumrand':                                                                                                                                                                                        
                steps = torch.rand((rays_d.shape[-2], N_samples), device=device) * stepsize * 2                                                                                                                    
                step = torch.cumsum(steps, dim=1)                                                                                                                                                                  

            case 'midpoint':                                                                                                                                                                                       
                step = stepsize * rng
        # steps = torch.rand((rays_d.shape[-2], N_samples), device=device) * stepsize * 2
        interpx = (t_min[..., None] + step)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        # add size
        rays_pts = torch.cat([rays_pts, interpx.unsqueeze(-1)/focal], dim=-1)
        env_mask = torch.zeros_like(mask_outbbox)
        env_mask[:, N_samples:] = 1

        if self.contract_space:
            mask_outbbox = torch.zeros_like(mask_outbbox)

        return rays_pts, interpx, ~mask_outbbox, env_mask

    def sample(self, rays_chunk, focal, ndc_ray=False, override_near=None, is_train=False, N_samples=-1):
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

        device = rays_chunk.device
        N, M = xyz_sampled.shape[:2]
        # sample alphas and cull samples from the ray
        # alpha_mask = self.alphaMask.sample_alpha(
        #     xyz_sampled[ray_valid], contract_space=self.contract_space)
        coords, cas = self.xyz2coords(xyz_sampled[ray_valid][..., :3])
        indices = raymarching.morton3D(coords).long() # [N]
        alpha = self.density_grid[cas, indices]
        alpha_mask = alpha > self.active_density_thresh

        alpha_mask = (self.density_bitfield[indices // 8] & (1 << (indices % 8))) > 0
        # if (~alpha_mask).sum() > 10:
        #     ic(alpha_mask.sum() / alpha_mask.numel(), alpha.mean())

        ray_invalid = ~ray_valid
        ray_invalid[ray_valid] |= (~alpha_mask)
        ray_valid = ~ray_invalid

        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        return xyz_sampled[ray_valid], ray_valid, M, z_vals, dists, torch.ones((N), dtype=bool, device=device)

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

    def xyz2coords(self, xyz):
        cas = 0
        cas_xyzs = xyz
        bound = min(2 ** cas, self.bound)
        half_grid_size = bound / self.grid_size

        o_xyzs = (cas_xyzs / (bound - half_grid_size)).clip(min=-1, max=1)
        coords = (o_xyzs+1) / 2 * (self.grid_size - 1)
        return coords.long(), cas

    def coords2xyz(self, coords, cas):
        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

        # cascading
        bound = min(2 ** cas, self.bound)
        half_grid_size = bound / self.grid_size
        # scale to current cascade's resolution
        cas_xyzs = xyzs * (bound - half_grid_size)
        # add noise in [-hgs, hgs]
        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
        # query density
        return cas_xyzs

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        if len(intrinsic) == 3:
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
        else:
            fx, fy, cx, cy = intrinsic
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # update count 
                            count[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[count == 0] = -1
        print(f'[mark untrained grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')

    def check_schedule(self, iteration, rf):
        if iteration % self.update_freq == 0:
            self.update(rf)
        return False

    def update(self, rf, decay=0.95, S=128, init=False):
        self.aabb = rf.aabb
        self.units = rf.units
        self.contract_space = rf.contract_space
        # reso_mask = reso_cur
        if not init:
            self.update_density(rf, decay, S=S)
            # self.grid_size = rf.grid_size
        self.nSamples = rf.nSamples*self.multiplier
        self.stepSize = rf.stepSize/self.multiplier
        # ic(self.nSamples, self.stepSize)

    @torch.no_grad()
    def update_density(self, rf, decay=0.95, S=128):
        start = time.time()
        # call before each epoch to update extra states.

        ### update density grid

        tmp_grid = -torch.ones_like(self.density_grid)
        
        # full update.
        if self.iter_density < 16:
        #if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                        # construct points
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]

                        # cascading
                        for cas in range(self.cascade):
                            cas_xyzs = self.coords2xyz(coords, cas)
                            # query density
                            cas_norm = rf.normalize_coord(cas_xyzs)
                            sigmas = rf.compute_densityfeature(cas_norm).reshape(-1)
                            # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                            # scale == 2 * sqrt(3) / 1024
                            # assign 
                            tmp_grid[cas, indices] = sigmas
                            # tmp_grid[cas, indices] = torch.exp(-sigmas)

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = self.grid_size ** 3 // 4 # H * H * H / 2
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_grid.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_grid.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                cas_xyzs = self.coords2xyz(coords, cas)
                # query density
                cas_norm = rf.normalize_coord(cas_xyzs)
                sigmas = rf.compute_densityfeature(cas_norm).reshape(-1)
                # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                # scale == 2 * sqrt(3) / 1024
                # assign 
                tmp_grid[cas, indices] = sigmas

        tmp_grid *= rf.distance_scale * self.stepsize
        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()
        self.iter_density += 1

        # convert to bitfield
        self.active_density_thresh = min(self.mean_density, self.threshold)
        self.density_bitfield = raymarching.packbits(self.density_grid, self.active_density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        # print(f'[density grid] {time.time()-start} min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, thresh={self.active_density_thresh} occ_rate={(self.density_grid > self.threshold).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')
