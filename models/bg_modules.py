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
    def __init__(self, bg_resolution=512, num_levels=1, featureC=128, activation='identity', power=4,
                 stds = [1, 2, 4, 8], betas=[0.9, 0.99], mipbias=+0.5, interp_pyramid=True, lr=0.15, mipbias_lr=1e-3, mipnoise=0.5, learnable_bias=True):
        super().__init__()
        self.num_levels = num_levels
        self.interp_pyramid = interp_pyramid
        self.activation = activation
        self.power = power
        self.align_corners = True
        self.smoothing = 1
        self.mipbias_lr = mipbias_lr
        self.lr = lr
        start_mip = self.num_levels - 1
        self.mipnoise = mipnoise
        self.betas = betas
        self.max_mip = start_mip
        if learnable_bias:
            self.register_parameter('mipbias', torch.nn.Parameter(torch.tensor(mipbias, dtype=float)))
        else:
            self.mipbias = mipbias

        self.stds = stds
        ic(self.stds)
        if len(self.stds) > 0:
            self.stds.sort(reverse=True)
            s = max(self.stds)*4-1
            blur_kernel = create_blur_pyramid(s, self.stds)
            ic(blur_kernel.shape)
            self.register_buffer('blur_kernel', blur_kernel)
            self.max_mip += math.log(2*max(self.stds)) / math.log(self.power)

        self.bg_mats = nn.ParameterList([
            # nn.Parameter(0.5 * torch.randn((1, 6, bg_resolution // self.power**i , bg_resolution // self.power**i, 3)) / (self.num_levels - i))
            # nn.Parameter(-0.5 * torch.ones((1, 6, bg_resolution // self.power**i , bg_resolution // self.power**i, 3)) / (self.num_levels - i))
            nn.Parameter(0.1 * torch.ones((1, 6, bg_resolution // self.power**i , bg_resolution // self.power**i, 3)) / (self.num_levels - i))
            for i in range(num_levels-1, -1, -1)])
        # self.activation_fn = torch.nn.Softplus(beta=3)

    def get_optparam_groups(self):
        return [
            {'params': self.bg_mats,
             'betas': self.betas,
             'lr': self.lr,
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
            return torch.exp(x-2).clip(min=0.01, max=1000)

    def calc_weight(self, mip):
        # return 1/2**(self.num_levels-mip)
        return 1#/(self.num_levels-mip)
        # return 1
        # return 1/(self.num_levels-mip)**2

    def calc_mip(self, i):
        return self.num_levels - 1 - i

    @property
    def bg_resolution(self):
        return self.bg_mats[-1].shape[2]

    def create_pyramid(self):
        start_mip = self.calc_mip(0)
        weight = self.calc_weight(start_mip)
        if self.interp_pyramid:
            bg_mat = 0
            bg_resolution = self.bg_mats[0].shape[2]
            # for i, mat in enumerate(self.bg_mats[:2]):
            for i, mat in enumerate(self.bg_mats):
                interp = F.interpolate(mat.squeeze(0).permute(0, 3, 1, 2), size=(bg_resolution, bg_resolution), mode='bilinear', align_corners=self.align_corners) / (i+2)
                bg_mat += interp
        else:
            bg_mat = self.bg_mats[0].squeeze(0).permute(0, 3, 1, 2) * weight
        bg_mat = self.activation_fn(bg_mat)
        convmats = cubemap_convolve(bg_mat, self.blur_kernel)
        # convmats = self.activation_fn(convmats)
        for i, s in enumerate(self.stds):
            mat = convmats[:, 3*i:3*i+3].permute(0, 2, 3, 1).unsqueeze(0)
            mip = start_mip + math.log(2*s) / math.log(self.power)
            yield mat, mip

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
        # for bg_mat, mip in self.iter_levels():
        #     bg_mat = torch.cat(bg_mat.data.unbind(1), dim=2).permute(0, 3, 1, 2)
            # bg_mat = F.interpolate(bg_mat, size=(self.bg_resolution, self.bg_resolution*6), mode='bilinear', align_corners=self.align_corners)
            # bg_mat = bg_mat * self.calc_weight(mip)
            # bg_mats += bg_mat
            # bg_mats = bg_mat
            im = self.activation_fn(bg_mat)
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
        # miplevel = miplevel.clip(0, self.max_mip-1)
        return miplevel.clip(0)
        
    def forward(self, viewdirs, saSample, max_level=None):
        # saSample is log of saSample
        B = viewdirs.shape[0]
        max_level = self.num_levels if max_level is None else max_level
        V = viewdirs.reshape(1, -1, 1, 3).contiguous()
        miplevel = self.sa2mip(viewdirs, saSample)
        # ic(miplevel.mean())

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

            # weight = (mip - miplevel + 1).clip(0, 1).reshape(-1, 1)
            # # embs.append(emb / 2**i * mask.reshape(-1, 1))
            # # embs.append(emb / (i+1) * weight + (1-weight) * offset)
            # sumemb += emb * self.calc_weight(mip) * weight
            sumemb += emb
            # embs.append(emb / 2**(i) * weight.reshape(-1, 1))
            if mip >= max_level:
                break
        img = self.activation_fn(sumemb)
        # if miplevel.max() >= self.num_levels-1 and len(self.stds) > 0:
        #     mask = miplevel.reshape(-1) >= self.num_levels-1
        #     blur_img = 0
        #     weights = 0
        #     for bg_mat, mip in self.create_pyramid():
        #         weight = 1 - (mip - miplevel.reshape(-1, 1)[mask]).abs().clip(0, 1)
        #         # ic(torch.cat([weight, miplevel.reshape(-1, 1)[mask]], dim=1), mip)
        #         emb = nvdr.texture(bg_mat.contiguous(), V[:, mask].contiguous(), boundary_mode='cube')
        #         emb = emb.reshape(-1, 3)
        #         blur_img += emb * weight
        #         weights += weight
        #     img[mask] = blur_img / (weights+1e-8)

        return img

class HierarchicalBG(torch.nn.Module):
    def __init__(self, bg_rank, unwrap_fn, bg_resolution=512, num_levels=2, featureC=128, activation='identity', num_layers=2, power=2, mipbias=0):
        super().__init__()
        self.num_levels = num_levels
        self.power = power
        self.mipbias = mipbias
        self.bg_mats = nn.ParameterList([
            # nn.Parameter(0.5 * torch.randn((1, 3, bg_resolution // self.power**i * unwrap_fn.H_mul, bg_resolution // self.power**i * unwrap_fn.W_mul)))
            nn.Parameter(0.5 * torch.rand((1, 3, bg_resolution // self.power**i * unwrap_fn.H_mul, bg_resolution // self.power**i * unwrap_fn.W_mul)))# / (self.num_levels-i) )
            # nn.Parameter(0.1 * torch.ones((1, 6, bg_resolution // self.power**i , bg_resolution // self.power**i, 3)))
            for i in range(num_levels-1, -1, -1)])

        self.unwrap_fn = unwrap_fn
        self.bg_rank = bg_rank
        if num_layers == 0 and bg_rank == 3:
            self.bg_net = torch.nn.Identity()
        else:
            self.bg_net = nn.Sequential(
                nn.Linear(bg_rank, featureC, bias=False),
                *sum([[
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(featureC, featureC, bias=False)
                    ] for _ in range(num_layers-2)], []),
                torch.nn.ReLU(inplace=True),
                nn.Linear(featureC, 3, bias=False),
            )
        self.align_corners = True
        self.smoothing = 1
        start_mip = self.num_levels - 1
        self.max_mip = start_mip

    def iter_levels(self):
        for i, bg_mat in enumerate(self.bg_mats):
            yield bg_mat, self.calc_mip(i)

    def activation_fn(self, x):
        # return F.softplus(x-10, beta=0.2)
        return F.softplus(x+0.5, beta=6)
        # return F.relu(x+0.5)

    def calc_weight(self, mip):
        # return 1/2**(self.num_levels-mip)
        # return 1/(self.num_levels-mip)
        return 1
        # return 1/(self.num_levels-mip)**2

    def calc_mip(self, i):
        return self.num_levels - 1 - i

    @property
    def bg_resolution(self):
        return self.bg_mats[-1].shape[2]

    @torch.no_grad()
    def save(self, path, prefix='', tonemap=None):
        bg_mats = 0
        bg_resolution = self.bg_mats[-1].shape[2] // self.unwrap_fn.H_mul
        for bg_mat, mip in self.iter_levels():
            bg_mat = F.interpolate(bg_mat.data, size=(bg_resolution*self.unwrap_fn.H_mul, bg_resolution*self.unwrap_fn.W_mul), mode='bilinear', align_corners=self.align_corners)
            bg_mat = bg_mat * self.calc_weight(mip)
            bg_mats += bg_mat
            # bg_mats = bg_mat
            im = self.activation_fn(bg_mats)
            if tonemap is not None:
                im = tonemap(im)
            else:
                im = im.clamp(0, 1)
            im = (255*im).short().permute(0, 2, 3, 1).squeeze(0)
            im = im.cpu().numpy()
            im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path / f'{prefix}pano{mip}.png'), im)

    def sa2mip(self, u, saSample):
        h, w = self.bg_mats[-1].shape[-2], self.bg_mats[-1].shape[-3]
        saTexel = 4 * math.pi / (6*h*w) * 4
        # TODO calculate distortion of cube map for saTexel
        # distortion = self.unwrap_fn.calc_distortion(u)
        saTexel = self.unwrap_fn.calc_distortion(u) / h / w
        # saTexel is the ratio to the solid angle subtended by one pixel of the 0th mipmap level
        num_pixels = self.bg_mats[-1].numel() // 3
        # saTexel = distortion / num_pixels
        miplevel = ((saSample - torch.log(saTexel)) / math.log(self.power))/2 + self.mipbias
        miplevel = miplevel.clip(0, self.max_mip-1)
        return miplevel

    @torch.no_grad()
    def upsample(self, bg_resolution):
        return
        
    def forward(self, viewdirs, saSample, max_level=None):
        B = viewdirs.shape[0]
        x = self.unwrap_fn(viewdirs)
        res = self.bg_mats[-1].shape[-2]
        max_level = self.num_levels if max_level is None else max_level
        miplevel = self.sa2mip(viewdirs, saSample)

        sumemb = 0
        for bg_mat, mip in self.iter_levels():
            # emb = F.grid_sample(smooth_mat.permute(1, 0, 2, 3), x, mode='bilinear', align_corners=self.align_corners)
            emb = F.grid_sample(bg_mat, x, mode='bilinear', align_corners=self.align_corners)
            emb = emb.reshape(self.bg_rank, -1).T
            # offset = bg_mat.mean()

            weight = (mip - miplevel + 1).clip(0, 1).reshape(-1, 1)
            # embs.append(emb / 2**i * mask.reshape(-1, 1))
            # embs.append(emb / (i+1) * weight + (1-weight) * offset)
            sumemb += emb * self.calc_weight(mip) * weight
            # embs.append(emb / 2**(i) * weight.reshape(-1, 1))
            if mip >= max_level:
                break
        img = self.activation_fn(sumemb)
        return img


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

