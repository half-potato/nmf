import torch
import math
import raymarching_full as raymarching 
import torch.nn.functional as F
from numba import jit
import numpy as np
from icecream import ic

class Raymarcher(torch.nn.Module):
    def __init__(self,
                 bound=2.0,
                 min_near=0.2,
                 density_thresh=0.002,
                 max_steps=1024,
                 max_samples=int(1.1e6),
                 dt_gamma=0,
                 grid_size=128,
                 perturb=False):
        super().__init__()

        self.bound = bound
        self.cascade = int(1 + math.ceil(math.log2(bound)))
        self.grid_size = grid_size
        # self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = grid_size
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.dt_gamma = dt_gamma
        self.max_steps = max_steps
        self.perturb = perturb
        self.max_samples = max_samples 
        self.stepsize = 0.003383
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


    def sample(self, rays_chunk, focal, ndc_ray=False, override_near=None, is_train=False, N_samples=-1):
        device = rays_chunk.device
        rays_o = rays_chunk[:, :3].contiguous().view(-1, 3)
        rays_d = rays_chunk[:, 3:6].contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact

        # aabb = self.aabb_train if is_train else self.aabb_infer
        aabb = self.aabb_train
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        # force_all_rays = not is_train
        force_all_rays = True
        counter = self.step_counter[self.local_step % 16]
        counter.zero_() # set to 0
        self.local_step += 1

        fxyzs, deltas, ray_valid = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade,
                self.grid_size, self.max_samples, nears, fars, counter, self.mean_count, self.perturb, -1, force_all_rays, self.dt_gamma, self.max_steps)
        ray_valid = ray_valid > 0
        whole_valid = torch.cumsum(ray_valid.sum(dim=1), dim=0) < self.max_samples
        # retained = torch.zeros_like(ray_valid)
        # i, j = torch.stack(torch.where(ray_valid), dim=0)[:, :self.max_samples]
        # # ic(i, j)
        # retained[i, j] = True
        # whole_valid = retained.sum(dim=1) == ray_valid.sum(dim=1)
        # ic(whole_valid.sum(), whole_valid2.sum())
        # ic(ray_valid.sum(dim=1), whole_valid)
        # ic(ray_valid.sum(dim=1)[whole_valid].float().mean(), ray_valid.sum(dim=1)[whole_valid2].float().mean())

        M = max(ray_valid.sum(dim=1).max(dim=0).values, 2)
        ray_valid = ray_valid[whole_valid, :] 
        fxyzs = fxyzs[whole_valid, :] 
        z_vals = deltas[whole_valid, :, 1]
        dists = deltas[whole_valid, :, 0]

        fxyzs = torch.cat([
            fxyzs,
            (z_vals / focal)[..., None]
        ], dim=-1)
        xyzs = fxyzs[ray_valid]
        M = fxyzs.shape[1]
        # ic(whole_valid.sum(), ray_valid.sum(dim=1).float().mean(), xyzs.shape)

        # print(self.density_bitfield.sum())
        # xyzs: (M, 4) values
        # dirs: the view direction. don't need this because we are packing it back up
        # delta: (M, 2). 0 = dt, 1 = z

        # rays: index, offset, num_steps
        # offset: index offset to the xyz samples for this ray
        # num_steps: number of samples for this ray

        # next, we want to convert rays to the ray_valid mask
        # M = rays.max(dim=0).values[2]
        # ray_valid = torch.as_tensor(rays2ray_valid(N, M, xyzs.shape[0], rays.detach().cpu().numpy()))
        # if not is_train:
        #     ic(ray_valid.sum(), xyzs.shape, rays.sum(dim=0), dirs.shape, deltas.shape, rays.shape, rays)
        # z_vals = torch.zeros((N, M), device=device)
        # dists = torch.zeros((N, M), device=device)
        # dists[ray_valid] = deltas[:, 0]
        # z_vals[ray_valid] = deltas[:, 1]

        # full_xyzs = torch.zeros((N, M, 3), device=device)
        # full_xyzs[ray_valid] = xyzs
        # ic(ray_valid.sum(dim=1)[0], torch.cat([full_xyzs[0], torch.arange(M, device=device)[:, None]], dim=-1), rays[0])

        # attach size
        # ic(deltas.max(dim=0), rays.max(dim=0))

        return xyzs, ray_valid, M, z_vals, dists, whole_valid

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
        if iteration % 16 == 0:
            self.update(rf)
        return False

    @torch.no_grad()
    def update(self, rf, decay=0.95, S=128, init=False):
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
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            cas_norm = rf.normalize_coord(cas_xyzs)
                            sigmas = rf.compute_densityfeature(cas_norm).reshape(-1)
                            # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                            # scale == 2 * sqrt(3) / 1024
                            sigmas *= rf.distance_scale * self.stepsize
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
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                cas_norm = rf.normalize_coord(cas_xyzs)
                sigmas = rf.compute_densityfeature(cas_norm).reshape(-1)
                # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                # scale == 2 * sqrt(3) / 1024
                sigmas *= rf.distance_scale * self.stepsize
                # assign 
                tmp_grid[cas, indices] = sigmas

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, thresh={density_thresh} occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')
