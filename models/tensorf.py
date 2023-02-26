import torch
from modules.pt_selectors import select_bounces
from mutils import normalize
from modules.row_mask_sum import row_mask_sum
from icecream import ic
import math
from modules import sh

class TensoRF(torch.nn.Module):
    def __init__(self, app_dim, diffuse_module):
        super().__init__()
        self.diffuse_module = diffuse_module(in_channels=app_dim)
        self.needs_normals = False
        self.outputs = {}

        self.mean_ratios = None
        self.max_retrace_rays = []

    def calibrate(self, args, *fargs):
        return args

    def get_optparam_groups(self, lr_scale=1):
        grad_vars = []
        grad_vars += [{'params': self.diffuse_module.parameters(),
                       'lr': self.diffuse_module.lr*lr_scale}]
        return grad_vars

    def check_schedule(self, iter, batch_mul, **kwargs):
        return False

    def update_n_samples(self, n_samples):
        return
        # assert(len(self.target_num_samples) == len(self.max_retrace_rays))
        if len(n_samples) == len(self.max_retrace_rays):
            ratios = [n_rays / n_sample if n_sample > 0 else None for n_rays, n_sample in zip(self.max_retrace_rays, n_samples)]
            if self.mean_ratios is None:
                self.mean_ratios = ratios
            else:
                self.mean_ratios = [
                        (min(0.1*ratio + 0.9*mean_ratio, 1, ratio) if ratio is not None else mean_ratio) if mean_ratio is not None else ratio
                        for ratio, mean_ratio in zip(ratios, self.mean_ratios)]
            self.max_retrace_rays = [
                    min(int(target * ratio + 1), maxv) if ratio is not None else prev
                    for target, ratio, maxv, prev in zip(self.target_num_samples, self.mean_ratios, self.max_brdf_rays[:-1], self.max_retrace_rays)]
            # ic(ratios, self.mean_ratios, n_samples, self.max_retrace_rays, self.target_num_samples)


    def forward(self, xyzs, xyzs_normed, app_features, viewdirs, normals, weights, app_mask, B, recur, render_reflection, bg_module, is_train, eps=torch.finfo(torch.float32).eps):
        # xyzs: (M, 4)
        # viewdirs: (M, 3)
        # normals: (M, 3)
        # weights: (M)
        # B: number of rays being cast
        # recur: recursion counter
        # render_reflection: function that casts out rays to allow for recursion
        debug = {}
        device = xyzs.device

        rgb = self.diffuse_module(
            xyzs_normed, viewdirs, app_features)
        return rgb, debug

