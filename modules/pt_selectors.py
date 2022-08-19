import torch
from icecream import ic

class Selector:
    def __init__(self, max_selected=99999999, val_thres=0, weight_thres=0, **kwargs):
        self.max_selected = max_selected
        self.val_thres = val_thres
        self.weight_thres = weight_thres

    def __call__(self, app_mask, weight, VdotL, val):
        # app_mask: (B, N) with M true elements
        # weight: (B, N)
        # val: (M)
        # bounce_mask: (M)
        bounce_mask = self._forward(app_mask, weight, VdotL, val)

        # derived masks
        full_bounce_mask = torch.zeros_like(app_mask)
        inv_full_bounce_mask = torch.zeros_like(app_mask)
        # combine the two masks because double masking causes issues
        ainds, ajinds = torch.where(app_mask)
        full_bounce_mask[ainds[bounce_mask], ajinds[bounce_mask]] = 1
        inv_full_bounce_mask[ainds[~bounce_mask], ajinds[~bounce_mask]] = 1
        return bounce_mask, full_bounce_mask, inv_full_bounce_mask

class TopNCombined(Selector):
    def _forward(self, app_mask, weight, VdotL, prob):
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
    def _forward(self, app_mask, weight, VdotL, prob):
        pweight = weight[app_mask]
        nmask = (VdotL.reshape(pweight.shape)) > 0
        bounce_mask = (pweight > self.weight_thres) & nmask
        if bounce_mask.sum() > self.max_selected:
            sweight, _ = torch.sort(-pweight[nmask].flatten())
            t = -sweight[self.max_selected]
            bounce_mask = pweight * nmask > t
        return bounce_mask

class TopNRoughness(Selector):
    def _forward(self, app_mask, weight, VdotL, prob):
        bounce_mask = prob > self.val_thres
        if bounce_mask.sum() > self.max_selected:
            sprob, _ = torch.sort(-prob.flatten())
            t = -sprob[self.max_selected]
            bounce_mask = prob > t
        return bounce_mask

class Surface(Selector):
    def __init__(self, max_selected=99999999, eps=1e-2, weight_thres=0, **kwargs):
        super().__init__(max_selected=max_selected, eps=eps, weight_thres=weight_thres, **kwargs)
        self.eps = eps

    def _forward(self, app_mask, weight, VdotL, prob):
        thres = torch.max(weight, dim=1, keepdim=True).values - self.eps
        ssort, _ = torch.sort(-weight.flatten())
        t = -ssort[self.max_selected]
        thres = thres.clip(min=max(self.weight_thres, t))
        imask = weight > thres
        return imask[app_mask]
