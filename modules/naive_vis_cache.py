import torch
from mutils import morton3D, normalize
from icecream import ic
import imageio
from pathlib import Path
import numpy as np
from loguru import logger

class NaiveVisCache(torch.nn.Module):
    def __init__(self, grid_size, jump=4, midpoint=128, required_count=0, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.midpoint = midpoint
        self.jump = jump
        self.init_count = 0
        self.required_count = required_count
        numer = torch.ones((self.grid_size, self.grid_size, self.grid_size, 6), dtype=torch.int)
        self.register_buffer('numer', numer)
        denom = torch.ones((self.grid_size, self.grid_size, self.grid_size, 6), dtype=torch.int)
        self.register_buffer('denom', denom)

    def is_initialized(self):
        return self.init_count > self.required_count

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

        coords = ((norm_ray_origins/2+0.5)*(self.grid_size-1)).clip(0, self.grid_size-1).long()
        return coords[..., 0], coords[..., 1], coords[..., 2], face_index

    @torch.no_grad()
    def forward(self, norm_ray_origins, viewdirs, *args, **kwargs):
        i, j, k, face_index = self.rays2inds(norm_ray_origins, viewdirs)
        n = self.numer[i, j, k, face_index]
        d = self.denom[i, j, k, face_index]
        # high if bg is visible
        return n/d

    def render(self):
        N = self.numer.shape[-1]
        ims = []
        for direction in range(N):
            volume = self.numer[..., direction].float() / self.denom[..., direction].float().clip(min=1)
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
    def fit(self, norm_ray_origins, viewdirs, bgvisibility):
        # visibility is a bool bgvisibility is true if the bg is visible
        mask = (norm_ray_origins.abs() < 1).all(dim=-1)
        i, j, k, face_index = self.rays2inds(norm_ray_origins[mask], viewdirs[mask])
        bgvisibility = bgvisibility.reshape(-1)[mask]
        eps = 5e-3
        self.init_count += 1
        # vals = self.cache[i, j, k, face_index].int() + torch.where(bgvisibility, 1, -self.jump)
        # vis_mask = ((norm_ray_origins[..., 0] < eps) & (norm_ray_origins[..., 1] > -eps)) & (norm_ray_origins.abs().min(dim=1).values < eps)
        # vals = self.cache[i, j, k, face_index].int() + torch.where(~vis_mask, 1, -self.jump)
        # vals = self.cache[indices, face_index].int() + torch.where(bgvisibility, self.jump, -self.jump)

        # ic((vis_mask | bgvisibility).sum(), (vis_mask & bgvisibility).sum())
        # ic((vis_mask).sum(), (bgvisibility).sum())
        # self.cache[i, j, k, face_index] = vals.clamp(0, 255).type(torch.uint8)
        self.numer[i, j, k, face_index] += bgvisibility.int()
        self.denom[i, j, k, face_index] += 1
