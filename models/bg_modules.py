import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from . import safemath
import math
import numpy as np
import cv2
from .render_modules import str2fn
import math
import nvdiffrast.torch as nvdr
from .cubemap_conv import cubemap_convolve, create_blur_pyramid
import matplotlib.pyplot as plt


class DualParaboloidUnwrap(torch.nn.Module):
    def __init__(self, b=1.2) -> None:
        # b is the scaling factor that adds to the edge
        super().__init__()
        self.H_mul = 2
        self.W_mul = 4
        self.b = b

    def forward(self, viewdirs):
        B = viewdirs.shape[0]
        x, y, z = viewdirs[:, 0:1], viewdirs[:, 1:2], viewdirs[:, 2:3]
        s = -torch.sign(z) * x/(1+z.abs())/self.b/2 + torch.where(z>0, 1/2, -1/2)
        t = -torch.sign(z) * y/(1+z.abs())/self.b

        x = torch.stack([ s, t ], dim=-1)
        return x.reshape(1, 1, -1, 2)

    def calc_distortion(self, u):
        return 4 * self.b**2 * (u[..., 2].abs()+1)**2

class PanoUnwrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.H_mul = 2
        self.W_mul = 4

    def forward(self, viewdirs):
        B = viewdirs.shape[0]
        a, b, c = viewdirs[:, 0:1], viewdirs[:, 1:2], viewdirs[:, 2:3]
        norm2d = torch.sqrt(a**2+b**2)
        phi = safemath.atan2(b, a)
        theta = safemath.atan2(c, norm2d)
        x = torch.cat([
            (phi % (2*np.pi) - np.pi) / np.pi,
            theta/np.pi*2,
        ], dim=1).reshape(1, 1, -1, 2)
        return x

class CubeUnwrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.H_mul = 1
        self.W_mul = 6

    def forward(self, viewdirs):
        B = viewdirs.shape[0]
        # mul = (1+1e-5)
        mul = 1
        l = 1/(1-1/64)
        sqdirs = mul * viewdirs / (torch.linalg.norm(viewdirs, dim=-1, keepdim=True, ord=torch.inf) + 1e-8)
        a, b, c = sqdirs[:, 0], sqdirs[:, 1], sqdirs[:, 2]
        # quadrants are -x, +x, +y, -y, +z, -z
        quadrants = [
            (a >=  mul, (b, c), 1),
            (a <= -mul, (b, c), 0),
            (b >=  mul, (a, c), 2),
            (b <= -mul, (a, c), 3),
            (c >=  mul, (a, b), 4),
            (c <= -mul, (a, b), 5),
        ]
        coords = torch.zeros_like(viewdirs[..., :2])
        for cond, (x, y), offset_mul in quadrants:
            coords[..., 0][cond] = ((x[cond] / l +1)/2 + offset_mul)/3 - 1
            coords[..., 1][cond] = y[cond] / l
        return coords.reshape(1, 1, -1, 2)


class HierarchicalCubeMap(torch.nn.Module):
    def __init__(self, bg_resolution=512, num_levels=1, featureC=128, activation='identity', power=4, brightness_lr=0.01, mul_lr=0.01,
                 stds = [1, 2, 4, 8], betas=[0.9, 0.99], mipbias=+0.5, init_val=-2, interp_pyramid=True, lr=0.15, mipbias_lr=1e-3, mipnoise=0.5, learnable_bias=True):
        super().__init__()
        self.num_levels = num_levels
        self.interp_pyramid = interp_pyramid
        self.activation = activation
        self.power = power
        self.align_corners = True
        self.smoothing = 1
        self.mipbias_lr = mipbias_lr
        self.lr = lr
        self.mul_lr = mul_lr
        start_mip = self.num_levels - 1
        self.mipnoise = mipnoise
        self.betas = betas
        self.max_mip = start_mip
        self.register_parameter('brightness', torch.nn.Parameter(torch.tensor(0.0, dtype=float)))
        self.register_parameter('mul', torch.nn.Parameter(torch.tensor(1.0, dtype=float)))
        if learnable_bias:
            self.register_parameter('mipbias', torch.nn.Parameter(torch.tensor(mipbias, dtype=float)))
        else:
            self.mipbias = mipbias
        self.brightness_lr = brightness_lr

        self.bg_mats = nn.ParameterList([
            nn.Parameter(init_val * torch.ones((1, 6, bg_resolution // self.power**i , bg_resolution // self.power**i, 3)) / (self.num_levels - i))
            for i in range(num_levels-1, -1, -1)])

    def get_optparam_groups(self):
        return [
            {'params': self.bg_mats,
             'betas': self.betas,
             'lr': self.lr,
             'name': 'bg'},
            {'params': self.brightness,
             'lr': self.brightness_lr,
             'name': 'bg'},
            {'params': self.mul,
             'lr': self.mul_lr,
             'betas': [0.9, 0.999],
             'name': 'bg'},
            {'params': [self.mipbias],
             'lr': self.mipbias_lr,
             'name': 'mipbias'}
        ]

    def activation_fn(self, x):
        if self.activation == 'softplus':
            return F.softplus(x, beta=6)
        elif self.activation == 'clip':
            return x.clip(min=1e-3)
        else:
            return torch.exp(x).clip(min=0.01, max=1000)

    def calc_weight(self, mip):
        return 1

    def calc_mip(self, i):
        return self.num_levels - 1 - i

    @property
    def bg_resolution(self):
        return self.bg_mats[-1].shape[2]

    def get_cubemap_faces(self):
        bg_mats = [[] for _ in range(6)]
        for stacked_bg_mat, mip in self.iter_levels():
            for i, bg_mat in enumerate(stacked_bg_mat.data.unbind(1)):
                bg_mat = F.interpolate(bg_mat.permute(0, 3, 1, 2), size=(self.bg_resolution, self.bg_resolution), mode='bilinear', align_corners=self.align_corners)
                bg_mat *= self.calc_weight(mip)
                bg_mats[i].append(bg_mat)
        return bg_mats

    def iter_levels(self):
        for i, bg_mat in enumerate(self.bg_mats):
            yield bg_mat, self.calc_mip(i)

    @torch.no_grad()
    def save(self, path, prefix='', tonemap=None):
        bg_mats = self.get_cubemap_faces()
        for i in range(self.num_levels):
            bg_mat = torch.cat([sum(bg_mats[j][:i+1]) for j in range(6)], dim=3)
            mip = self.calc_mip(i)
            im = self.activation_fn(self.brightness.clip(min=-1, max=2) + self.mul*bg_mat)
            if tonemap is not None:
                im = tonemap(im)
            im = im.clamp(0, 1)
            im = (255*im).short().permute(0, 2, 3, 1).squeeze(0)
            im = im.cpu().numpy()
            im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path / f'{prefix}pano{mip}.png'), im)

    @torch.no_grad()
    def upsample(self, bg_resolution):
        return

    def sa2mip(self, u, saSample):
        h, w = self.bg_mats[-1].shape[-2], self.bg_mats[-1].shape[-3]
        saTexel = 4 * math.pi / (6*h*w) * 4
        # TODO calculate distortion of cube map for saTexel
        # distortion = 4 * math.pi / 6
        distortion = 1/torch.linalg.norm(u, dim=-1, ord=torch.inf).reshape(*saSample.shape)
        saTexel = distortion / h / w
        # saTexel is the ratio to the solid angle subtended by one pixel of the 0th mipmap level
        num_pixels = self.bg_mats[-1].numel() // 3
        # saTexel = distortion / num_pixels
        miplevel = ((saSample - torch.log(saTexel)) / math.log(self.power))/2 + self.mipbias + self.mipnoise * torch.rand_like(saSample)
        return miplevel.clip(0)
        
    def forward(self, viewdirs, saSample, max_level=None):
        # saSample is log of saSample
        B = viewdirs.shape[0]
        max_level = self.num_levels if max_level is None else max_level
        V = viewdirs.reshape(1, -1, 1, 3).contiguous()
        miplevel = self.sa2mip(viewdirs, saSample)

        sumemb = 0
        for bg_mat, mip in self.iter_levels():
            # emb = F.grid_sample(smooth_mat.permute(1, 0, 2, 3), x, mode='bilinear', align_corners=self.align_corners)
            # emb = nvdr.texture(bg_mat, V, uv_da=torch.exp(saSample).reshape(1, -1, 1, 1).expand(1, -1, 1, 6).contiguous(), boundary_mode='cube')
                    # mip_level_bias=mipbias*torch.ones((1, B, 1), device=V.device))
            emb = nvdr.texture(bg_mat, V, boundary_mode='cube', mip_level_bias=miplevel.reshape(1, -1, 1))
            # emb = nvdr.texture(bg_mat, V, boundary_mode='cube', mip_level_bias=2*torch.ones((1, B, 1), device=V.device))
            # emb = nvdr.texture(bg_mat, V, boundary_mode='cube', mip_level_bias=miplevel)
            emb = emb.reshape(-1, 3)
            # offset = bg_mat.mean()

            sumemb += emb
            if mip >= max_level:
                break
        img = self.activation_fn(self.brightness.clip(min=-1, max=2) + self.mul*sumemb)
        # ic(img.mean(), self.brightness, self.mul)

        return img

