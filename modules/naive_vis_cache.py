import torch
from mutils import morton3D, normalize

class NaiveVisCache(torch.nn.Module):
    def __init__(self, grid_size, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.midpoint = 128
        self.jump = 8
        cache = (self.midpoint) * torch.ones((self.grid_size**3, 6), dtype=torch.uint8)
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

        coords = ((norm_ray_origins/2+0.5)*self.grid_size).clip(0, self.grid_size-1)
        indices = morton3D(coords.long())
        return indices, face_index

    @torch.no_grad()
    def mask(self, norm_ray_origins, viewdirs, world_bounces, *args, **kwargs):
        indices, face_index = self.rays2inds(norm_ray_origins, -viewdirs)
        vals = self.cache[indices, face_index]
        # inds = torch.argsort(vals)[:world_bounces]
        # mask = torch.zeros_like(vals, dtype=torch.bool)
        # mask[inds] = True
        # vals is high when it reaches BG and low when it does not
        mask = vals < self.midpoint
        return mask

    @torch.no_grad()
    def forward(self, norm_ray_origins, viewdirs, *args, **kwargs):
        indices, face_index = self.rays2inds(norm_ray_origins, viewdirs)
        vals = self.cache[indices, face_index]
        return vals > self.midpoint

    @torch.no_grad()
    def update(self, norm_ray_origins, viewdirs, app_features, termination, bgvisibility):
        # visibility is a bool
        indices, face_index = self.rays2inds(norm_ray_origins, viewdirs)
        vals = self.cache[indices, face_index].int() + torch.where(bgvisibility, 0, -self.jump)
        self.cache[indices, face_index] = vals.clamp(0, 255).type(torch.uint8)
        return torch.tensor(0.0, device=norm_ray_origins.device)
