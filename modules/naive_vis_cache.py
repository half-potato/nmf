import torch
from mutils import morton3D, normalize
from icecream import ic
import imageio
from pathlib import Path
import numpy as np

class NaiveVisCache(torch.nn.Module):
    def __init__(self, grid_size, bound, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.bound = bound
        self.midpoint = 128
        self.jump = 20
        cache = (self.midpoint) * torch.ones((self.grid_size, self.grid_size, self.grid_size, 6), dtype=torch.uint8)
        self.register_buffer('cache', cache)

    @torch.no_grad()
    def rays2inds(self, norm_ray_origins, viewdirs):
        # convert viewdirs to index
        B = viewdirs.shape[0]
        mul = 1
        sqdirs = mul * normalize(viewdirs, ord=torch.inf)
        face_index = torch.zeros((B), dtype=torch.long, device=norm_ray_origins.device)
        a, b, c = sqdirs[:, 0], sqdirs[:, 1], sqdirs[:, 2]
        # quadrants are -x, +x, +y, -y, +z, -z
        quadrants = [
            a >=  mul,
            a <= -mul,
            b >=  mul,
            b <= -mul,
            c >=  mul,
            c <= -mul,
        ]
        for i, cond in enumerate(quadrants):
            face_index[cond] = i

        # coords = ((norm_ray_origins/self.bound/2+0.5)*(self.grid_size-1))
        coords = ((norm_ray_origins/2+0.5)*(self.grid_size-1)).clip(0, self.grid_size-1).long()
        # coords = ((norm_ray_origins/2+0.5)*(self.grid_size-1)).long()
        # coords = #
        # ic(coords.max(), coords.min())
        # indices = morton3D(coords.long())
        # ic(coords, indices, self.cache.shape)
        # ic(indices.max(), indices.min(), self.cache.shape)
        return coords[..., 0], coords[..., 1], coords[..., 2], face_index

    @torch.no_grad()
    def mask(self, norm_ray_origins, viewdirs, world_bounces, full_bounce_mask, ray_mask, weight, *args, **kwargs):
        i, j, k, face_index = self.rays2inds(norm_ray_origins, -viewdirs)
        vals = self.cache[i, j, k, face_index]
        mask = torch.zeros_like(full_bounce_mask)
        mask[range(mask.shape[0]), weight.max(dim=1).indices] = True
        max_weight_mask = mask[full_bounce_mask].reshape(-1, 1).expand(*ray_mask.shape)[ray_mask]
        vals[~max_weight_mask] = 255

        # select a maximum of one point along each ray to allow for world bounces
        
        # eps = 2e-2
        # vis_mask = ((norm_ray_origins[..., 0] < eps) & (norm_ray_origins[..., 1] > -eps))
        inds = vals.argsort()[:world_bounces]
        # inds = torch.argsort(vals)[:world_bounces]
        # mask = torch.zeros_like(vals, dtype=torch.bool)
        # mask[inds] = True
        # vals is high when it reaches BG and low when it does not
        # ic(vals.min(), vals.float().mean())
        mask1 = vals < self.midpoint
        mask2 = torch.zeros_like(mask1)
        mask2[inds] = True
        mask = mask1 & mask2
        # mask = mask1 & max_weight_mask
        # ic((vis_mask | mask).sum(), (vis_mask & mask).sum())
        # ic((vis_mask).sum(), (mask).sum())
        return mask

    @torch.no_grad()
    def forward(self, norm_ray_origins, viewdirs, *args, **kwargs):
        i, j, k, face_index = self.rays2inds(norm_ray_origins, viewdirs)
        vals = self.cache[i, j, k, face_index]
        return vals > self.midpoint

    @torch.no_grad()
    def ray_update(self, viewdirs, xyz_normed, app_mask, ray_valid):
        # xyz_sampled: (M, 4) float. premasked valid sample points
        # viewdirs: (M, 3) float. premasked corresponding viewdirs
        # max_samps = N
        # z_vals: (b, N) float. distance along ray to sample
        # ray_valid: (b, N) bool. mask of which samples are valid
        # app_mask: [b, N]

        # convert app_mask into bgvisibility
        # this can be done by computing the termination for each ray, then calculating the distance to each point along the ray
        inds = torch.arange(app_mask.shape[1], device=viewdirs.device)
        term_ind = (app_mask * inds.reshape(1, -1)).min(dim=1, keepdim=True).values
        bgvisibility = (inds < term_ind)[ray_valid]
        self.update(xyz_normed[..., :3], -viewdirs, bgvisibility)

    def render(self):
        N = self.cache.shape[-1]
        ims = []
        for direction in range(N):
            volume = self.cache[..., direction].float()
            im1 = volume.mean(dim=0)
            im2 = volume.mean(dim=1)
            im3 = volume.mean(dim=2)
            ims.append(torch.cat([im1, im2, im3], dim=0).cpu())
        im = torch.cat(ims, dim=1)
        return im

    def save(self, path, prefix=''):
        im = self.render()
        im = im.cpu().numpy().astype(np.uint8)
        imageio.imwrite(str(Path(path) / f'{prefix}_viscache.png'), im)

    @torch.no_grad()
    def update(self, norm_ray_origins, viewdirs, bgvisibility):
        # visibility is a bool bgvisibility is true if the bg is visible
        # output mask is true if bg is not visible
        i, j, k, face_index = self.rays2inds(norm_ray_origins, viewdirs)
        eps = 5e-3
        # vals = self.cache[i, j, k, face_index].int() + torch.where(bgvisibility, 1, -self.jump)
        vis_mask = ((norm_ray_origins[..., 0] < eps) & (norm_ray_origins[..., 1] > -eps))
        vals = self.cache[i, j, k, face_index].int() + torch.where(~vis_mask, 0, -self.jump)
        # vals = self.cache[indices, face_index].int() + torch.where(bgvisibility, self.jump, -self.jump)

        # ic((vis_mask | bgvisibility).sum(), (vis_mask & bgvisibility).sum())
        # ic((vis_mask).sum(), (bgvisibility).sum())
        self.cache[i, j, k, face_index] = vals.clamp(0, 255).type(torch.uint8)
