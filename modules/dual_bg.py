import torch

class MultiBG(torch.nn.Module):
    def __init__(self, bgs):
        super().__init__()
        self.bgs = torch.nn.ModuleList(bgs)
        self.bg_index = 0

    def get_optparam_groups(self, lr_scale=1):
        # lr_scale = 1
        return sum([bg.get_optparam_groups(lr_scale) for bg in self.bgs], [])

    def hw(self):
        return self.bgs[self.bg_index].hw()

    def get_device(self):
        return self.bgs[self.bg_index].get_device()

    def activation_fn(self, x):
        return self.bgs[self.bg_index].activation_fn(x)

    def get_brightness(self):
        return self.bgs[self.bg_index].get_brightness()

    @property
    def mipbias(self):
        return self.bgs[self.bg_index].mipbias

    @property
    def bg_resolution(self):
        return self.bgs[self.bg_index].bg_resolution

    def mean_color(self):
        return self.bgs[self.bg_index].mean_color()

    @torch.no_grad()
    def calc_envmap_psnr(self, gt_im, fH=500):
        return self.bgs[self.bg_index].calc_envmap_psnr(gt_im, fH)


    def get_spherical_harmonics(self, G, mipval=-5):
        return self.bgs[self.bg_index].get_spherical_harmonics(G, mipval)

    @torch.no_grad()
    def save(self, path, prefix='', tonemap=None):
        for ind, bg in enumerate(self.bgs):
            bg.save(path, f"{prefix}_{ind}", tonemap=tonemap)

    def sa2mip(self, u, saSample, eps=torch.finfo(torch.float32).eps):
        return self.bgs[self.bg_index].sa2mip(u, saSample, eps)

    def tv_loss(self):
        return self.bgs[self.bg_index].tv_loss()

    def forward(self, viewdirs, saSample, max_level=None):
        return self.bgs[self.bg_index].forward(viewdirs, saSample, max_level)

