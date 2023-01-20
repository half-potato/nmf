import torch
from icecream import ic

@torch.no_grad()
def select_bounces(weights, app_mask, num_roughness_rays, percent_bright):
    device = weights.device

    # pt_limit = weights * num_roughness_rays + 0.5
    pt_limit = weights / (weights.sum().clip(min=1e-3)) * num_roughness_rays + 0.5
    
    # nopt_mask = pt_limit.max(dim=1).values < 1
    # pt_limit[nopt_mask] = pt_limit[nopt_mask] / pt_limit.max(dim=1, keepdim=True).values.clamp(min=0.9, max=1)[nopt_mask]# + 0.01
    pt_limit = pt_limit[app_mask]

    num_samples = pt_limit.floor().quantile(0.999).clip(max=500).int()
    if num_samples == 0:
        pt_limit = torch.where(torch.rand_like(pt_limit) < num_roughness_rays / pt_limit.shape[0], 1.1, 0)
        num_samples = 1

    # create ray_mask
    ray_mask = torch.arange(num_samples, device=device).reshape(1, -1) < pt_limit.reshape(-1, 1).floor()
    bright_limit = pt_limit.reshape(-1, 1)*(1-percent_bright)
    main_ray_mask = torch.arange(num_samples, device=device).reshape(1, -1) < bright_limit.floor()
    bright_mask = ray_mask & ~main_ray_mask

    bounce_mask = ray_mask.sum(dim=-1) > 0
    ray_mask = ray_mask[bounce_mask]
    bright_mask = bright_mask[bounce_mask]
    # ic(pt_limit.floor().max(), num_samples, ray_mask.sum(), ray_mask.shape, num_roughness_rays)

    return bounce_mask, ray_mask, bright_mask

"""
@torch.no_grad()
def select_bounces(app_mask, weight, num_roughness_rays, percent_bright):
    # app_mask: (B, N) with M true elements
    # weight: (B, N)
    # val: (M)
    # bounce_mask: (M)
    device = weight.device

    # mweight = weight.clone()
    # # mweight[~app_mask] = 0
    # # mweight[inv_full_bounce_mask] = 0
    # # ic(mweight.sum(dim=1, keepdim=True))
    # n_weight = weight / mweight.sum(dim=1, keepdim=True).clip(min=0.1)
    n_weight = weight.clone()
    # n_weight[app_mask] = n_weight[app_mask] * (VdotN.reshape(-1) < 0)
    pt_limit = n_weight * num_roughness_rays + 0.5
    nopt_mask = pt_limit.max(dim=1).values < 1
    pt_limit[nopt_mask] = pt_limit[nopt_mask] / pt_limit.max(dim=1, keepdim=True).values.clamp(min=0.1, max=1)[nopt_mask]# + 0.01
    # num_missing = (num_roughness_rays - pt_limit.int().sum(dim=1, keepdim=True))
    # b = num_missing - torch.rand_like(weight) * app_mask.sum(dim=1, keepdim=True)
    # pt_limit[full_bounce_mask] += (b < 0)[full_bounce_mask]

    num_samples = pt_limit.floor().quantile(0.999).int()
    ray_mask = torch.arange(num_samples, device=device).reshape(1, -1) < pt_limit[app_mask].reshape(-1, 1).floor()
    bright_limit = pt_limit[app_mask].reshape(-1, 1)*(1-percent_bright)
    main_ray_mask = torch.arange(num_samples, device=device).reshape(1, -1) < bright_limit.floor()
    bright_mask = ray_mask & ~main_ray_mask
    # ic(1, ray_mask.sum())
    # ray_mask = torch.arange(num_roughness_rays, device=device).reshape(1, -1) < pt_limit[app_mask].reshape(-1, 1)-1
    # ic(2, ray_mask.sum())

    bounce_mask = ray_mask.sum(dim=-1) > 0
    ray_mask = ray_mask[bounce_mask]
    bright_mask = bright_mask[bounce_mask]

    # derived masks
    full_bounce_mask = torch.zeros_like(app_mask)
    inv_full_bounce_mask = torch.zeros_like(app_mask)
    ainds, ajinds = torch.where(app_mask)
    full_bounce_mask[ainds[bounce_mask], ajinds[bounce_mask]] = 1
    inv_full_bounce_mask[ainds[~bounce_mask], ajinds[~bounce_mask]] = 1

    return bounce_mask, full_bounce_mask, inv_full_bounce_mask, ray_mask, bright_mask
"""
