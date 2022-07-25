import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from . import safemath
import numpy as np
import cv2
from .render_modules import str2fn
import math
import nvdiffrast.torch as nvdr


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
    def __init__(self, bg_rank, bg_resolution=512, num_levels=2, featureC=128, activation='identity', num_layers=0, power=4):
        super().__init__()
        self.bg_resolution = bg_resolution
        self.num_levels = num_levels
        self.power = power
        self.bg_mats = nn.ParameterList([
            # nn.Parameter(0.5 * torch.rand((1, bg_rank, self.power**i * bg_resolution*unwrap_fn.H_mul, self.power**i * bg_resolution*unwrap_fn.W_mul)))
            nn.Parameter(0.1 * torch.ones((1, 6, self.power**i * bg_resolution, self.power**i * bg_resolution, bg_rank)))
            for i in range(num_levels)])
        self.bg_rank = bg_rank
        activation_fn = str2fn(activation)
        if num_layers == 0 and bg_rank == 3:
            self.bg_net = activation_fn
        else:
            self.bg_net = nn.Sequential(
                nn.Linear(bg_rank, featureC, bias=False),
                *sum([[
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(featureC, featureC, bias=False)
                    ] for _ in range(num_layers-2)], []),
                torch.nn.ReLU(inplace=True),
                nn.Linear(featureC, 3, bias=False),
                activation_fn
            )
        self.align_corners = False
        self.smoothing = 1

    @torch.no_grad()
    def save(self, path):
        bg_resolution = self.bg_mats[-1].shape[2]
        bg_mats = 0
        for i in range(self.num_levels):
            bg_mat = torch.cat(self.bg_mats[i].data.unbind(1), dim=2).permute(0, 3, 1, 2)
            bg_mat = F.interpolate(bg_mat, size=(bg_resolution, bg_resolution*6), mode='bilinear', align_corners=self.align_corners)
            bg_mat = bg_mat / (i+1)
            bg_mats += bg_mat
            im = (255*(self.bg_net(bg_mats)).clamp(0, 1)).short().permute(0, 2, 3, 1).squeeze(0)
            im = im.cpu().numpy()
            im = im[::-1].astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path / f'pano{i}.png'), im)

    @torch.no_grad()
    def upsample(self, bg_resolution):
        return
        
    def forward(self, viewdirs, saSample, max_level=None):
        B = viewdirs.shape[0]
        max_level = self.num_levels if max_level is None else max_level
        res = self.bg_mats[-1].shape[-2]
        saTexel = 4 * math.pi / (6*res*res)
        # miplevel = (torch.log(saSample / saTexel) / math.log(self.power) / self.power).clip(0, self.num_levels-5)
        # mip level is 0 when it wants the full image and inf when it wants just the color
        miplevel = (torch.log(saSample / saTexel) / math.log(self.power) / 2).clip(0, max_level-1)

        embs = []
        for i, bg_mat in enumerate(self.bg_mats):

            # smooth_kern = gkern(2*int(self.smoothing)+1, std=self.smoothing+1e-8, device=viewdirs.device)
            # s = smooth_kern.shape[-1]
            # smooth_mat = F.conv2d(bg_mat.permute(1, 0, 2, 3), smooth_kern.reshape(1, -1, s, s), stride=1, padding=s//2)
            #
            # emb = F.grid_sample(smooth_mat.permute(1, 0, 2, 3), x, mode='bilinear', align_corners=self.align_corners)
            emb = nvdr.texture(bg_mat, viewdirs.reshape(1, -1, 1, 3).contiguous(), boundary_mode='cube')
            emb = emb.reshape(-1, self.bg_rank)
            # offset = bg_mat.mean()

            weight = (self.num_levels - miplevel - i + 1).clip(0, 1).reshape(-1, 1)
            # ic(weight, miplevel, miplevel-i, (self.num_levels - miplevel - i))
            # weight = torch.sigmoid((1-roughness)*max_level - i - 2)
            # embs.append(emb / 2**i * mask.reshape(-1, 1))
            # embs.append(emb / (i+1) * weight + (1-weight) * offset)
            embs.append(emb / (i+1) * weight)
            # embs.append(emb / 2**(i) * weight.reshape(-1, 1))
            if i >= max_level:
                break
        emb = sum(embs)
        return self.bg_net(emb)

class HierarchicalBG(torch.nn.Module):
    def __init__(self, bg_rank, unwrap_fn, bg_resolution=512, num_levels=2, featureC=128, activation='identity', num_layers=2, power=4):
        super().__init__()
        self.bg_resolution = bg_resolution
        self.num_levels = num_levels
        self.power = power
        self.bg_mats = nn.ParameterList([
            # nn.Parameter(0.5 * torch.rand((1, bg_rank, self.power**i * bg_resolution*unwrap_fn.H_mul, self.power**i * bg_resolution*unwrap_fn.W_mul)))
            nn.Parameter(0.1 * torch.ones((1, bg_rank, self.power**i * bg_resolution*unwrap_fn.H_mul, self.power**i * bg_resolution*unwrap_fn.W_mul)))
            for i in range(num_levels)])
        self.unwrap_fn = unwrap_fn
        self.bg_rank = bg_rank
        activation_fn = str2fn(activation)
        if num_layers == 0 and bg_rank == 3:
            self.bg_net = activation_fn
        else:
            self.bg_net = nn.Sequential(
                nn.Linear(bg_rank, featureC, bias=False),
                *sum([[
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(featureC, featureC, bias=False)
                    ] for _ in range(num_layers-2)], []),
                torch.nn.ReLU(inplace=True),
                nn.Linear(featureC, 3, bias=False),
                activation_fn
            )
        self.align_corners = False
        self.smoothing = 1

    @torch.no_grad()
    def save(self, path):
        bg_resolution = self.bg_mats[-1].shape[2] // self.unwrap_fn.H_mul
        bg_mats = []
        for i in range(self.num_levels):
            bg_mat = F.interpolate(self.bg_mats[i].data, size=(bg_resolution*self.unwrap_fn.H_mul, bg_resolution*self.unwrap_fn.W_mul), mode='bilinear', align_corners=self.align_corners)
            bg_mat = bg_mat / (i+1)
            bg_mats.append(bg_mat)
        bg_mat = sum(bg_mats)
        im = (255*(self.bg_net(bg_mat)).clamp(0, 1)).short().permute(0, 2, 3, 1).squeeze(0)
        im = im.cpu().numpy()
        im = im[::-1].astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, im)

    @torch.no_grad()
    def upsample(self, bg_resolution):
        return
        
    def forward(self, viewdirs, saSample, max_level=None):
        B = viewdirs.shape[0]
        x = self.unwrap_fn(viewdirs)
        max_level = self.num_levels if max_level is None else max_level
        res = self.bg_mats[-1].shape[-2]
        saTexel = 4 * math.pi / (6*res*res)
        miplevel = (torch.log(saSample / saTexel) / math.log(self.power) / self.power).clip(1, self.num_levels)
        # ic(miplevel.max(), miplevel.mean())
        # miplevel = torch.zeros_like(miplevel) + 4
        # ic(miplevel.min(), miplevel.max(), saSample.min(), saSample.max())
        # mip level is 0 when it wants the full image and inf when it wants just the color

        embs = []
        for i, bg_mat in enumerate(self.bg_mats):

            # smooth_kern = gkern(2*int(self.smoothing)+1, std=self.smoothing+1e-8, device=viewdirs.device)
            # s = smooth_kern.shape[-1]
            # smooth_mat = F.conv2d(bg_mat.permute(1, 0, 2, 3), smooth_kern.reshape(1, -1, s, s), stride=1, padding=s//2)
            #
            # emb = F.grid_sample(smooth_mat.permute(1, 0, 2, 3), x, mode='bilinear', align_corners=self.align_corners)

            emb = F.grid_sample(bg_mat, x, mode='bicubic', align_corners=self.align_corners)
            emb = emb.reshape(self.bg_rank, -1).T
            weight = (self.num_levels - miplevel - i).clip(0, 1)
            # ic(weight, miplevel, miplevel-i, (self.num_levels - miplevel - i))
            # weight = torch.sigmoid((1-roughness)*max_level - i - 2)
            # embs.append(emb / 2**i * mask.reshape(-1, 1))
            embs.append(emb / (i+1) * weight.reshape(-1, 1))
            # embs.append(emb / 2**(i) * weight.reshape(-1, 1))
            if i >= max_level:
                break
        emb = sum(embs)
        return self.bg_net(emb)


class BackgroundRender(torch.nn.Module):
    def __init__(self, bg_rank, unwrap_fn, bg_resolution=512, view_encoder=None, featureC=128, num_layers=2):
        super().__init__()
        self.bg_mat = nn.Parameter(0.1 * torch.randn((1, bg_rank, bg_resolution*unwrap_fn.H_mul, bg_resolution*unwrap_fn.W_mul))) # [1, R, H, W]
        self.bg_resolution = bg_resolution
        self.view_encoder = view_encoder
        self.unwrap_fn = unwrap_fn
        self.bg_rank = bg_rank
        d = self.view_encoder.dim() if self.view_encoder is not None else 0
        if num_layers == 0 and bg_rank == 3:
            # TODO REMOVE
            # self.bg_net = nn.Softplus()
            self.bg_net = nn.Identity()
        else:
            self.bg_net = nn.Sequential(
                nn.Linear(bg_rank+d, featureC, bias=False),
                *sum([[
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(featureC, featureC, bias=False)
                    ] for _ in range(num_layers-2)], []),
                torch.nn.ReLU(inplace=True),
                nn.Linear(featureC, 3, bias=False),
                nn.Softplus()
            )
        self.align_corners = False

    @torch.no_grad()
    def save(self, path):
        im = (255*self.bg_net(self.bg_mat).clamp(0, 1)).short().permute(0, 2, 3, 1).squeeze(0)
        im = im.cpu().numpy()
        im = im[::-1].astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, im)

    @torch.no_grad()
    def upsample(self, bg_resolution):
        self.bg_resolution = bg_resolution
        self.bg_mat = torch.nn.Parameter(
            F.interpolate(self.bg_mat.data, size=(bg_resolution*self.unwrap_fn.H_mul, bg_resolution*self.unwrap_fn.W_mul), mode='bilinear', align_corners=self.align_corners)
        )
        
    def forward(self, viewdirs, roughness):
        B = viewdirs.shape[0]
        x = self.unwrap_fn(viewdirs)

        # col = ((x[..., 0]/2 +0.5)* self.bg_mat.shape[3]).long()
        # row = ((x[..., 1]/2 +0.5)* self.bg_mat.shape[2]).long()
        # ic(self.bg_mat[:, :, row, col])
        emb = F.grid_sample(self.bg_mat, x, mode='bilinear', align_corners=self.align_corners)
        emb = emb.reshape(self.bg_rank, -1).T
        # return torch.sigmoid(emb)
        if self.view_encoder is not None:
            shemb = self.view_encoder(viewdirs, torch.tensor(20.0, device=viewdirs.device)).reshape(B, -1)
            emb = torch.cat([emb, shemb], dim=1)
        return self.bg_net(emb)

