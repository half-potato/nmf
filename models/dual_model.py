import torch
from icecream import ic


class DualModel(torch.nn.Module):
    def __init__(self, appdim, model1, model2, warmup_iters, alternating, **kwargs):
        super().__init__()
        self.model1 = model1(appdim, **kwargs)
        self.model2 = model2(appdim, **kwargs)
        self.select1 = True
        self.warmup_iters = warmup_iters
        self.alternating = alternating

    def calibrate(self, config, *args, **kwargs):
        config = self.model1.calibrate(config, *args, save_config=False, **kwargs)
        config = self.model2.calibrate(config, *args, save_config=False, **kwargs)
        return config

    def get_optparam_groups(self, lr_scale=1):
        return self.model1.get_optparam_groups(
            lr_scale
        ) + self.model2.get_optparam_groups(lr_scale)

    def get_model(self, recur, is_train=True):
        return (
            self.model1 if self.select1 else (self.model1 if recur > 0 else self.model2)
        )

    def graph_brdfs(self, xyzs, viewdirs, app_features, res):
        return self.model2.graph_brdfs(xyzs, viewdirs, app_features, res)

    def check_schedule(self, iter, batch_mul, **kwargs):
        update = self.model1.check_schedule(iter, batch_mul, **kwargs)
        if iter > self.warmup_iters or (self.alternating and iter % 2 != 0):
            update |= self.model2.check_schedule(
                iter - self.warmup_iters, batch_mul, **kwargs
            )
            self.select1 = False
        else:
            self.select1 = True
        return update

    @property
    def max_retrace_rays(self):
        return self.get_model(0).max_retrace_rays

    def needs_normals(self, recur):
        # ic(self.get_model(recur), self.get_model(recur).needs_normals(recur), recur)
        return self.get_model(recur).needs_normals(recur)

    @property
    def outputs(self):
        return self.get_model(0).outputs

    def update_n_samples(self, n_samples):
        self.get_model(0).update_n_samples(n_samples)

    def forward(self, *args, is_train, recur, **kwargs):
        out = self.get_model(recur, is_train).forward(
            *args, is_train=is_train, recur=recur, **kwargs
        )
        return out
