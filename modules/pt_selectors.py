import torch
from icecream import ic

class Selector:
    def __init__(self, percent_bright=0, max_selected=99999999, val_thres=0, weight_thres=0, **kwargs):
        self.max_selected = max_selected
        self.val_thres = val_thres
        self.weight_thres = weight_thres
        self.percent_bright = percent_bright

    def __call__(self, app_mask, weight, VdotN, val, num_roughness_rays):
        # app_mask: (B, N) with M true elements
        # weight: (B, N)
        # val: (M)
        # bounce_mask: (M)
        device = weight.device

        # mweight = weight.clone()
        # # mweight[~app_mask] = 0
        # mweight[weight < self.val_thres] = 0
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
        bright_limit = pt_limit[app_mask].reshape(-1, 1)*(1-self.percent_bright)
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
