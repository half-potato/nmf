import torch
from icecream import ic

class MultiSampler:
    def __init__(self, sampler1, sampler2, max_samples):
        self.sampler1 = sampler1(max_samples=max_samples)
        self.sampler2 = sampler2(max_samples=max_samples)

    def sample(self, viewdir, normal, r1, r2, ray_mask, proportion, eps=torch.finfo(torch.float32).eps):
        device = viewdir.device
        device = ray_mask.device
        # r1 = torch.ones_like(r2)*0.04
        # r2 = torch.ones_like(r2)*0.01
        # proportion = torch.ones_like(proportion)*0.5

        ray_mask1 = torch.arange(ray_mask.shape[1], device=device).reshape(1, -1) < (ray_mask.sum(dim=1, keepdim=True) * proportion.reshape(-1, 1))
        ray_mask2 = ray_mask & ~ray_mask1
        L = torch.zeros((ray_mask.sum(), 3), device=device)
        row_world_basis = torch.zeros((ray_mask.sum(), 3, 3), device=device)

        L1, row_world_basis1 = self.sampler1.sample(
                    viewdir, normal,
                    r1, r1, ray_mask1)
        # row_world_basis needs to be the same
        L2, row_world_basis2 = self.sampler1.sample(
                    viewdir, normal,
                    r2, r2, ray_mask2)
        L[ray_mask1[ray_mask]] = L1
        L[ray_mask2[ray_mask]] = L2
        row_world_basis[ray_mask1[ray_mask]] = row_world_basis1
        row_world_basis[ray_mask2[ray_mask]] = row_world_basis2
        return L, row_world_basis

    def update(self, *args, **kwargs):
        self.sampler1.update(*args, **kwargs)
        self.sampler2.update(*args, **kwargs)

    def compute_prob(self, halfvec, eN, r1, r2, proportion, **kwargs):
        p1 = self.sampler1.compute_prob(halfvec, eN, r1, r1)
        p2 = self.sampler2.compute_prob(halfvec, eN, r2, r2)
        return proportion*p1 + (1-proportion)*p2

    def calculate_mipval(self, H, V, N, ray_mask, r1, r2, proportion, eps=torch.finfo(torch.float32).eps, **kwargs):
        # ray_mask1, ray_mask2 = self.split_ray_mask(ray_mask, proportion)
        mipval1 = self.sampler1.calculate_mipval(H, V, N, ray_mask, r1, proportion, eps=eps, **kwargs)
        mipval2 = self.sampler2.calculate_mipval(H, V, N, ray_mask, r2, proportion, eps=eps, **kwargs)
        # ic(mipval1.shape, mipval2.shape, proportion.shape)
        return proportion.reshape(-1, 1)*mipval1.reshape(-1, 1) + (1-proportion.reshape(-1, 1))*mipval2.reshape(-1, 1)
