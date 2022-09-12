import torch
from mutils import morton3D, normalize
from icecream import ic

class NaiveVisCache(torch.nn.Module):
    def __init__(self, grid_size, bound, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.bound = bound
        self.midpoint = 128
        self.jump = 8
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
    def mask(self, norm_ray_origins, viewdirs, world_bounces, *args, **kwargs):
        i, j, k, face_index = self.rays2inds(norm_ray_origins, -viewdirs)
        vals = self.cache[i, j, k, face_index]
        # eps = 2e-2
        # vis_mask = ((norm_ray_origins[..., 0] < eps) & (norm_ray_origins[..., 1] > -eps))
        inds = vals.argsort()[:world_bounces]
        # inds = torch.argsort(vals)[:world_bounces]
        # mask = torch.zeros_like(vals, dtype=torch.bool)
        # mask[inds] = True
        # vals is high when it reaches BG and low when it does not
        mask1 = vals < self.midpoint
        mask2 = torch.zeros_like(mask1)
        mask2[inds] = True
        mask = mask1 & mask2
        # ic((vis_mask | mask).sum(), (vis_mask & mask).sum())
        # ic((vis_mask).sum(), (mask).sum())
        return mask

    @torch.no_grad()
    def forward(self, norm_ray_origins, viewdirs, *args, **kwargs):
        i, j, k, face_index = self.rays2inds(norm_ray_origins, viewdirs)
        vals = self.cache[i, j, k, face_index]
        return vals > self.midpoint

    @torch.no_grad()
    def update(self, norm_ray_origins, viewdirs, app_features, termination, bgvisibility):
        # visibility is a bool
        # bgvisibility is true if the bg is visible
        # output mask is true if bg is not visible
        i, j, k, face_index = self.rays2inds(norm_ray_origins, viewdirs)
        eps = 2e-2
        vals = self.cache[i, j, k, face_index].int() + torch.where(bgvisibility, 0, -self.jump)
        # vals = self.cache[i, j, k, face_index].int() + torch.where(~vis_mask, 0, -self.jump)
        # vals = self.cache[indices, face_index].int() + torch.where(bgvisibility, self.jump, -self.jump)

        # ic((vis_mask | bgvisibility).sum(), (vis_mask & bgvisibility).sum())
        # ic((vis_mask).sum(), (bgvisibility).sum())
        self.cache[i, j, k, face_index] = vals.clamp(0, 255).type(torch.uint8)
        return torch.tensor(0.0, device=norm_ray_origins.device)
