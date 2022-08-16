import torch
import torch.nn.functional as F
from icecream import ic

class AlphaGridMask(torch.nn.Module):
    def __init__(self, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.register_buffer('aabb', aabb)

        aabbSize = self.aabb[1] - self.aabb[0]
        invgrid_size = 1.0/aabbSize * 2
        grid_size = torch.LongTensor(
            [alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]])
        self.register_buffer('grid_size', grid_size)
        self.register_buffer('invgrid_size', invgrid_size)
        self.register_buffer('alpha_volume', alpha_volume)

    def sample_alpha(self, xyz_sampled, contract_space=False):
        xyz_sampled = self.normalize_coord(xyz_sampled, contract_space)
        H, W, D = self.alpha_volume.shape
        i = ((xyz_sampled[..., 0]/2+0.5)*(H-1)).long()
        j = ((xyz_sampled[..., 1]/2+0.5)*(W-1)).long()
        k = ((xyz_sampled[..., 2]/2+0.5)*(D-1)).long()
        alpha_vals = self.alpha_volume[i, j, k]
        # alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled[..., :3].view(
        #     1, -1, 1, 1, 3), align_corners=False).view(-1)

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

class AlphaGridSampler:
    def __init__(self, enable_alpha_mask=False, near_far=[2, 6], nEnvSamples=0, update_list=[]):
        self.enable_alpha_mask = enable_alpha_mask
        self.alphaMask = None
        self.nEnvSamples = nEnvSamples
        self.near_far = near_far
        self.update_list = update_list

    def check_schedule(self, iteration, rf):
        if iteration in self.update_list:
            self.update(rf)
        return False

    def update(self, rf):
        # self.nSamples = rf.nSamples//8
        # self.stepSize = rf.stepSize*8
        # self.nSamples = rf.nSamples*16
        # self.stepSize = rf.stepSize/16
        self.nSamples = rf.nSamples
        self.stepSize = rf.stepSize

        self.aabb = rf.aabb
        self.units = rf.units
        self.contract_space = rf.contract_space
        self.grid_size = rf.grid_size
        # reso_mask = reso_cur
        # new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
        # if iteration == update_AlphaMask_list[0]:
        #     apply_correction = not torch.all(tensorf.alphaMask.grid_size == tensorf.rf.grid_size)
        #     rf.shrink(new_aabb, apply_correction)

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
        N_env_samples = N_env_samples if N_env_samples > 0 else self.nEnvSamples
        device = rays_o.device
        stepsize = self.stepSize
        near, far = self.near_far
        if override_near is not None:
            near = override_near
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1].to(rays_o) - rays_o) / vec
        rate_b = (self.aabb[0].to(rays_o) - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_min = near * torch.ones_like(t_min)

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
        steps = torch.rand((rays_d.shape[-2], N_samples), device=device) * stepsize * 2
        step = torch.cumsum(steps, dim=1)
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

    @torch.no_grad()
    def getDenseAlpha(self, grid_size=None):
        grid_size = self.grid_size if grid_size is None else grid_size

        dense_xyz = torch.stack([*torch.meshgrid(
            torch.linspace(-1, 1, grid_size[0]),
            torch.linspace(-1, 1, grid_size[1]),
            torch.linspace(-1, 1, grid_size[2])),
            torch.ones((grid_size[0], grid_size[1],
                       grid_size[2]))*self.units.min().cpu()*0.5
        ], -1).to(self.device)

        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(grid_size[0]):
            xyz_norm = dense_xyz[i].view(-1, 4)
            sigma_feature = self.compute_densityfeature(xyz_norm)
            sigma = self.feature2density(sigma_feature)
            alpha[i] = 1 - torch.exp(-sigma*self.stepSize).reshape(*alpha[i].shape)

        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, grid_size=(200, 200, 200)):

        alpha, dense_xyz = self.getDenseAlpha(grid_size)

        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = grid_size[0] * grid_size[1] * grid_size[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks,
                             padding=ks // 2, stride=1).view(grid_size[::-1])
        # alpha[alpha >= self.alphaMask_thres] = 1
        # alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.aabb, alpha > self.alphaMask_thres).to(self.device)

        valid_xyz = dense_xyz[alpha > 0.0]
        if valid_xyz.shape[0] < 1:
            print("No volume")
            return self.aabb

        xyz_min = valid_xyz.amin(0)[:3]
        xyz_max = valid_xyz.amax(0)[:3]

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" %
              (total/total_voxels*100))
        return new_aabb


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
        alpha_mask = torch.zeros((M), device=device, dtype=bool)
        if self.alphaMask is not None and self.enable_alpha_mask:
            alpha_mask[ray_valid] = self.alphaMask.sample_alpha(
                xyz_sampled[ray_valid], contract_space=self.contract_space)

            # T = torch.cumprod(torch.cat([
            #     torch.ones(alphas.shape[0], 1, device=alphas.device),
            #     1. - alphas + 1e-10
            # ], dim=-1), dim=-1)[:, :-1]
            # ray_invalid = ~ray_valid
            # ray_invalid |= (~alpha_mask)
            ray_valid ^= alpha_mask

        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        return xyz_sampled[ray_valid], ray_valid, M, z_vals, dists, torch.ones((N), dtype=bool, device=device)
