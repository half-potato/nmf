import torch
from icecream import ic

class Selector:
    def __init__(self, percent_bright=0, bounces_per_ray=10, max_selected=99999999, val_thres=0, weight_thres=0, **kwargs):
        self.max_selected = max_selected
        self.val_thres = val_thres
        self.weight_thres = weight_thres
        self.bounces_per_ray = bounces_per_ray
        self.percent_bright = percent_bright

    def __call__(self, app_mask, weight, VdotN, val, num_roughness_rays):
        # app_mask: (B, N) with M true elements
        # weight: (B, N)
        # val: (M)
        # bounce_mask: (M)
        bounce_mask = self._forward(app_mask, weight, VdotN, val)
        device = weight.device

        full_bounce_mask = torch.zeros_like(app_mask)
        ainds, ajinds = torch.where(app_mask)
        full_bounce_mask[ainds[bounce_mask], ajinds[bounce_mask]] = 1

        # mweight = weight.clone()
        # # mweight[~app_mask] = 0
        # mweight[weight < self.val_thres] = 0
        # # mweight[inv_full_bounce_mask] = 0
        # # ic(mweight.sum(dim=1, keepdim=True))
        # n_weight = weight / mweight.sum(dim=1, keepdim=True).clip(min=0.1)
        n_weight = weight
        pt_limit = n_weight * self.bounces_per_ray + 0.5
        nopt_mask = pt_limit.max(dim=1).values < 1
        pt_limit[nopt_mask] = pt_limit[nopt_mask] / pt_limit.max(dim=1, keepdim=True).values.clamp(min=0.1, max=1)[nopt_mask]# + 0.01
        # num_missing = (self.bounces_per_ray - pt_limit.int().sum(dim=1, keepdim=True))
        # b = num_missing - torch.rand_like(weight) * app_mask.sum(dim=1, keepdim=True)
        # pt_limit[full_bounce_mask] += (b < 0)[full_bounce_mask]

        ray_mask = torch.arange(num_roughness_rays, device=device).reshape(1, -1) < pt_limit[app_mask].reshape(-1, 1).floor()
        bright_limit = pt_limit[app_mask].reshape(-1, 1)*(1-self.percent_bright)
        main_ray_mask = torch.arange(num_roughness_rays, device=device).reshape(1, -1) < bright_limit.floor()
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

class TopNCombined(Selector):
    def _forward(self, app_mask, weight, VdotN, prob):
        # N: int max number of selected
        # t: threshold

        pweight = weight[app_mask]
        # topmask = weight > (weight.max(dim=1, keepdim=True).values - wt)
        # ptopmask = topmask[app_mask]
        prob = prob.reshape(-1)# * ptopmask
        M = prob.shape[0]
        bounce_mask = torch.zeros((M), dtype=bool, device=app_mask.device)

        n_bounces = min(min(self.max_selected, M), ((prob > self.val_thres) & (pweight > self.weight_thres)).sum())
        inds = torch.argsort(-pweight*prob)[:n_bounces]
        bounce_mask[inds] = 1
        return bounce_mask

class TopNWeight(Selector):
    def _forward(self, app_mask, weight, VdotN, prob):
        pweight = weight[app_mask]
        nmask = (VdotN.reshape(pweight.shape)) > -0.0
        bounce_mask = (pweight > self.weight_thres) & nmask
        # if bounce_mask.sum() > self.max_selected:
        #     ic("HI")
        #     sweight, _ = torch.sort(-pweight[nmask].flatten())
        #     t = -sweight[self.max_selected]
        #     bounce_mask = pweight * nmask > t
        return bounce_mask

class TopNRoughness(Selector):
    def _forward(self, app_mask, weight, VdotN, prob):
        bounce_mask = prob > self.val_thres
        if bounce_mask.sum() > self.max_selected:
            sprob, _ = torch.sort(-prob.flatten())
            t = -sprob[self.max_selected]
            bounce_mask = prob > t
        return bounce_mask

class Surface(Selector):
    def __init__(self, bounces_per_ray, max_selected=99999999, eps=1e-2, weight_thres=0, **kwargs):
        super().__init__(max_selected=max_selected, eps=eps, weight_thres=weight_thres, **kwargs)
        self.eps = eps
        self.bounces_per_ray = bounces_per_ray

    def _forward(self, app_mask, weight, VdotN, prob):
        thres = torch.max(weight, dim=1, keepdim=True).values - self.eps
        ssort, _ = torch.sort(-weight.flatten())
        t = -ssort[self.max_selected]
        thres = thres.clip(min=max(self.weight_thres, t))
        imask = weight > thres
        return imask[app_mask]
