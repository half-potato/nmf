import torch
import bisect
from icecream import ic

class BrightnessSampler(torch.nn.Module):
    def __init__(self, num_samples, n, sampler, decay=0.99):
        super().__init__()
        self.n = n
        self.decay = decay
        self.sampler = sampler(num_samples)

        cache = torch.zeros((3*n, 5))
        self.cache_init = False
        self.register_buffer('cache', cache)

    def update(self, V, mipval, incoming_light):
        self.sampler.update(V, mipval, incoming_light)
        self.cache[:, 4] *= self.decay
        brightness = incoming_light.max(dim=-1).values

        minind = self.cache[:, 4].argmin()
        minval = self.cache[minind, 4]
        brightness = brightness.reshape(self.n, -1)
        mipval = mipval.reshape(self.n, -1)
        V = V.reshape(self.n, -1, 3)
        # todo this should have room for other bright spots
        # it should pick some optimal spread somehow
        for i, bs in enumerate(brightness):
            mi = bs.argmax()
            mv = bs[mi]
            if mv < minval:
                continue
            self.cache[minind, 4] = mv
            self.cache[minind, :3] = V[i, mi]
            self.cache[minind, 3] = mipval[i, mi]

            minind = self.cache[:, 4].argmin()
            minval = self.cache[minind, 4]
        self.cache_init = True

    def sample(self, num_samples, refdirs, viewdir, normal, roughness):
        n = self.n if self.cache_init else 0
        device = normal.device
        B = refdirs.shape[0]
        samp_rays, samp_mipval = self.sampler.sample(num_samples-n, refdirs, viewdir, normal, roughness)
        if n > 0:
            with torch.no_grad():
                cache_rays = self.cache[:, :3].reshape(1, -1, 3)
                cache_mipval = self.cache[:, 3].reshape(1, -1)

                # pick rays that are outward facing
                hemi = (normal.reshape(-1, 1, 3) * cache_rays).sum(dim=-1)
                sim = (refdirs.reshape(-1, 1, 3) * cache_rays).sum(dim=-1)
                score = torch.where(hemi > 0, -sim, torch.tensor(0.0, device=device))
                inds = torch.argsort(score, dim=1) < n
                # ic(inds.shape)
                sel_rays = cache_rays.expand(B, -1, 3)[inds].reshape(B, n, 3)
                sel_mipval = cache_mipval.expand(B, -1)[inds].reshape(B, n)
            # ic(samp_rays.shape, sel_rays.shape, sel_mipval.shape, cache_rays.expand(B, -1, 3).shape, self.cache.shape, samp_mipval.shape)
            rays = torch.cat([samp_rays, sel_rays], dim=1)
            mipval = torch.cat([samp_mipval, sel_mipval], dim=1)
        else:
            rays = samp_rays
            mipval = samp_mipval
        return rays, mipval
