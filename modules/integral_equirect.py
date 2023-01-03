import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import safemath
from icecream import ic
import numpy as np
import imageio
import cv2

class IntegralEquirect(torch.nn.Module):
    def __init__(self, bg_resolution, init_val, activation='identity',
                 mipbias=0, mipnoise=0,
                 lr=0.15, mipbias_lr=1e-3, brightness_lr=0.01, mul_lr=0.01,
                 mul_betas=[0.9, 0.999], betas=[0.9, 0.99]):
        super().__init__()
        data = init_val * torch.ones((1, 3, bg_resolution, 2*bg_resolution))
        self.bg_mat = nn.Parameter(data)
        # self.register_buffer('cache', cache)
        self.mipnoise = mipnoise
        # self.interp_mode = 'nearest'
        self.align_corners = False
        self.interp_mode = 'bilinear'
        self.register_parameter('mipbias', torch.nn.Parameter(torch.tensor(mipbias, dtype=float)))
        self.register_parameter('brightness', torch.nn.Parameter(torch.tensor(0.0, dtype=float)))
        self.register_parameter('mul', torch.nn.Parameter(torch.tensor(1.0, dtype=float)))

        self.lr = lr
        self.mul_lr = mul_lr
        self.mipbias_lr = mipbias_lr
        self.brightness_lr = brightness_lr

        self.mul_betas = mul_betas
        self.betas = betas

        self.activation = activation

    def get_optparam_groups(self, lr_scale=1):
        # lr_scale = 1
        return [
            {'params': self.bg_mat,
             'betas': self.betas,
             'lr': self.lr*lr_scale,
             'name': 'bg'},
            {'params': self.brightness,
             'lr': self.brightness_lr*lr_scale,
             'name': 'bg'},
            {'params': self.mul,
             'lr': self.mul_lr*lr_scale,
             'betas': self.mul_betas,
             'name': 'bg'},
            {'params': [self.mipbias],
             'lr': self.mipbias_lr*lr_scale,
             'name': 'mipbias'}
        ]

    def hw(self):
        h, w = self.bg_mat.shape[-1], self.bg_mat.shape[-2]
        return h, w

    def activation_fn(self, x):
        # x = torch.exp(self.brightness.clip(max=2)) + x
        x = self.brightness + self.mul*x
        if self.activation == 'softplus':
            return F.softplus(x, beta=6)
        elif self.activation == 'clip':
            return x.clip(min=1e-3)
        elif self.activation == 'identity':
            return x
        else:
            return torch.exp(x.clip(max=20))

    def get_device(self):
        return self.bg_mat.device

    @property
    def bg_resolution(self):
        return self.hw()[0]

    def mean_color(self):
        return self.activation_fn(self.bg_mat).reshape(-1, 3).mean(dim=0)

    @torch.no_grad()
    def save(self, path, prefix='', tonemap=None):
        im = self.activation_fn(self.bg_mat)
        if tonemap is not None:
            im = tonemap(im)
        im = im.clamp(0, 1)
        im = (255*im).short().permute(0, 2, 3, 1).squeeze(0)
        im = im.cpu().numpy()
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
        imageio.imwrite(str(path / f'{prefix}pano.png'), im)

    def sa2mip(self, u, saSample, eps=torch.finfo(torch.float32).eps):
        h, w = self.hw()
        saTexel = 4 * math.pi / (6*h*w) * 4
        # TODO calculate distortion of cube map for saTexel
        # distortion = 4 * math.pi / 6
        distortion_w = 1-u[:, 2]**2
        distortion_h = torch.ones_like(distortion_w)
        saSample = saSample.reshape(-1)

        saTexel_w = distortion_w / h / w
        saTexel_h = distortion_h / h / w
        # saTexel is the ratio to the solid angle subtended by one pixel of the 0th mipmap level
        # num_pixels = self.bg_mat.numel() // 3
        # saTexel = distortion / num_pixels
        miplevel_w = ((saSample - torch.log(saTexel_w.clip(min=eps))) / math.log(2))/2 + self.mipbias + self.mipnoise * torch.rand_like(saSample)
        miplevel_h = ((saSample - torch.log(saTexel_h.clip(min=eps))) / math.log(2))/2 + self.mipbias + self.mipnoise * torch.rand_like(saSample)
        
        return miplevel_w.clip(0), miplevel_h.clip(0)

    def forward(self, viewdirs, saSample, max_level=None):
        # Compute width and height of sample
        device = viewdirs.device
        eps = torch.finfo(torch.float32).eps
        miplevel_w, miplevel_h = self.sa2mip(viewdirs, saSample)
        h, w = self.hw()
        sw = 2 ** miplevel_w / w
        sh = 2 ** miplevel_h / h
        # sw = 0.20*torch.ones_like(sw)
        # sh = 0.20*torch.ones_like(sh)
        offset = torch.stack([sw, sh], dim=-1).reshape(1, 1, -1, 2)

        cum_mat = torch.cumsum(torch.cumsum(self.activation_fn(self.bg_mat), dim=2), dim=3)
        # cum_mat = torch.cumsum(self.bg_mat, dim=2)
        # cum_mat = torch.cumsum(self.bg_mat, dim=3)
        # cum_mat = self.bg_mat
        # ic(cum_mat)
        size = (offset * torch.tensor([w, h], device=device).reshape(1, 1, 1, 2)).prod(dim=-1).reshape(-1, 1)

        def integrate_area(bl, br, tl, tr, size):
            exceed_t = (tl[..., 1] - 1).clip(min=0)
            exceed_b = -(bl[..., 1] + 1).clip(max=0)
            # if exceed top, add to bot
            bl[..., 1] += exceed_t
            br[..., 1] += exceed_t
            tl[..., 1] += exceed_b
            tr[..., 1] += exceed_b

            bl = bl.clip(min=-1, max=1)
            br = br.clip(min=-1, max=1)
            tl = tl.clip(min=-1, max=1)
            tr = tr.clip(min=-1, max=1)
            blv = F.grid_sample(cum_mat, bl, mode=self.interp_mode, align_corners=self.align_corners, padding_mode='border')
            brv = F.grid_sample(cum_mat, br, mode=self.interp_mode, align_corners=self.align_corners, padding_mode='border')
            tlv = F.grid_sample(cum_mat, tl, mode=self.interp_mode, align_corners=self.align_corners, padding_mode='border')
            trv = F.grid_sample(cum_mat, tr, mode=self.interp_mode, align_corners=self.align_corners, padding_mode='border')
            # norm_size = (tr - bl).abs().clip(min=eps)
            # size = (norm_size * torch.tensor([w, h], device=device).reshape(1, 1, 1, 2)).prod(dim=-1).reshape(-1, 1)
            return (trv + blv - tlv - brv).reshape(3, -1).T / size

        # compute location of sample
        a, b, c = viewdirs[:, 0:1], viewdirs[:, 1:2], viewdirs[:, 2:3]
        norm2d = torch.sqrt(a**2+b**2)
        phi = safemath.atan2(b, a)
        theta = safemath.atan2(c, norm2d)
        coords = torch.cat([
            (phi % (2*math.pi) - math.pi) / math.pi,
            -theta/math.pi*2,
        ], dim=1)
        x = coords.reshape(1, 1, -1, 2)

        # bottom left is top left of image because image space = (-1, -1)
        bl = x - offset / 2
        tr = x + offset / 2
        br = x + torch.stack([sw, -sh], dim=-1).reshape(1, 1, -1, 2) / 2
        tl = x + torch.stack([-sw, sh], dim=-1).reshape(1, 1, -1, 2) / 2
        # for such a simple rectangular shape, it must retain a certain height near the poles, even with clipping
        # to achieve this, half the amount clipped off the top or bottom should be added back to the opposite side
        bg_vals = integrate_area(bl, br, tl, tr, size)

        exceed_r = tr[..., 0] > 1

        # set left points to left edge of pano
        bl_r = bl[exceed_r].reshape(1, 1, -1, 2)
        bl_r[..., 0] = -torch.ones_like(bl_r[..., 0])
        tl_r = tl[exceed_r].reshape(1, 1, -1, 2)
        tl_r[..., 0] = -torch.ones_like(bl_r[..., 0])
        tr_r = tr[exceed_r].reshape(1, 1, -1, 2)
        tr_r[..., 0] = tr_r[..., 0] - 2
        br_r = br[exceed_r].reshape(1, 1, -1, 2)
        br_r[..., 0] = br_r[..., 0] - 2

        rect_r = integrate_area(bl_r, br_r, tl_r, tr_r, size[exceed_r.reshape(-1)])
        bg_vals[exceed_r.reshape(-1)] += rect_r

        exceed_l = bl[..., 0] < -1
        # set left points to left edge of pano
        bl_l = bl[exceed_l].reshape(1, 1, -1, 2)
        bl_l[..., 0] = bl_l[..., 0] + 2
        tl_l = tl[exceed_l].reshape(1, 1, -1, 2)
        tl_l[..., 0] = tl_l[..., 0] + 2
        tr_l = tr[exceed_l].reshape(1, 1, -1, 2)
        tr_l[..., 0] = torch.ones_like(tr_l[..., 0])
        br_l = br[exceed_l].reshape(1, 1, -1, 2)
        br_l[..., 0] = torch.ones_like(br_l[..., 0])

        rect_l = integrate_area(bl_l, br_l, tl_l, tr_l, size[exceed_l.reshape(-1)])
        bg_vals[exceed_l.reshape(-1)] += rect_l

        # handle top and bottom
        cutoff = 1 - 2 / h
        top_row = self.bg_mat[..., 0, :].mean(dim=-1)
        bot_row = self.bg_mat[..., -1, :].mean(dim=-1)
        bg_vals[coords[:, 1] > cutoff] = bot_row
        bg_vals[coords[:, 1] < -cutoff] = top_row
        # return self.activation_fn(bg_vals)
        return bg_vals

if __name__ == "__main__":
    import imageio
    img = imageio.imread("backgrounds/ninomaru_teien_4k.exr")
    device = torch.device('cuda')
    data = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)/255
    new_w = 100
    data = F.interpolate(data, (new_w, 2*new_w))
    bg_res = data.shape[-2]
    # data = data[:100, :100]
    bg_module = IntegralEquirect(bg_res, 0, activation='exp')
    bg_module.bg_mat = nn.Parameter(data.log())

    bg_module = bg_module.to(device)
    res = bg_res
    ele_grid, azi_grid = torch.meshgrid(
        torch.linspace(math.pi/2, -math.pi/2, res, dtype=torch.float32),
        torch.linspace(0, 2*math.pi, 2*res, dtype=torch.float32), indexing='ij')
    # azi_grid = azi_grid - math.pi
    # each col of x ranges from -pi/2 to pi/2
    # each row of y ranges from -pi to pi
    ang_vecs = torch.stack([
        torch.cos(ele_grid) * torch.cos(azi_grid),
        torch.cos(ele_grid) * torch.sin(azi_grid),
        -torch.sin(ele_grid),
    ], dim=-1).reshape(-1, 3).to(device)
    sa_sample = -1*torch.ones((ang_vecs.shape[0], 1), device=device)
    vals = bg_module(ang_vecs, sa_sample)
    blur_img = vals.reshape(res, 2*res, 3)#.abs()
    imageio.imwrite("bg_blur.exr", blur_img.detach().cpu().numpy())
    imageio.imwrite("bg_blur_abs.exr", blur_img.abs().detach().cpu().numpy())
    imageio.imwrite("bg_noblur.exr", bg_module.bg_mat.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
