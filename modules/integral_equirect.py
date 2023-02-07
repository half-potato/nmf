import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import safemath
from icecream import ic
import numpy as np
import imageio
import cv2
from modules import sh
from sklearn import linear_model
import traceback

def integrate_area(bl, br, tl, tr, size, cum_mat, interp_mode='bilinear', align_corners=True):
    bl = bl.clip(min=-1, max=1)
    br = br.clip(min=-1, max=1)
    tl = tl.clip(min=-1, max=1)
    tr = tr.clip(min=-1, max=1)
    blv = F.grid_sample(cum_mat, bl, mode=interp_mode, align_corners=align_corners, padding_mode='zeros')
    brv = F.grid_sample(cum_mat, br, mode=interp_mode, align_corners=align_corners, padding_mode='zeros')
    tlv = F.grid_sample(cum_mat, tl, mode=interp_mode, align_corners=align_corners, padding_mode='zeros')
    trv = F.grid_sample(cum_mat, tr, mode=interp_mode, align_corners=align_corners, padding_mode='zeros')
    # norm_size = (tr - bl).abs().clip(min=eps)
    # size = (norm_size * torch.tensor([w, h], device=device).reshape(1, 1, 1, 2)).prod(dim=-1).reshape(-1, 1)
    return (trv + blv - tlv - brv).reshape(3, -1).T / size

def integrate_area_wrap_lr(bl, br, tl, tr, size, cum_mat, interp_mode='bilinear', align_corners=True):
    bg_vals = integrate_area(bl, br, tl, tr, size, cum_mat, interp_mode, align_corners)

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

    rect_r = integrate_area(bl_r, br_r, tl_r, tr_r, size[exceed_r.reshape(-1)], cum_mat, interp_mode, align_corners)
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

    rect_l = integrate_area(bl_l, br_l, tl_l, tr_l, size[exceed_l.reshape(-1)], cum_mat, interp_mode, align_corners)
    bg_vals[exceed_l.reshape(-1)] += rect_l
    return bg_vals

def integrate_area_wrap(bl, br, tl, tr, size, cum_mat, interp_mode='bilinear', align_corners=True):
    bg_vals = integrate_area_wrap_lr(bl, br, tl, tr, size,
                                     cum_mat, interp_mode=interp_mode, align_corners=align_corners)

    exceed_t = tl[..., 1] > 1
    # set top to be the top of the pano
    # set bot to be top of pano - overhang
    tl_t = tl[exceed_t].reshape(1, 1, -1, 2)
    rot_val_t = torch.where(tl_t[..., 0] > 0, -1, 1)
    overhang_t = (tl_t[..., 1] - 1).clip(max=0.5, min=0)
    tl_t[..., 1] = torch.ones_like(tl_t[..., 0])
    tl_t[..., 0] = tl_t[..., 0] + rot_val_t
    tr_t = tr[exceed_t].reshape(1, 1, -1, 2)
    tr_t[..., 1] = torch.ones_like(tr_t[..., 0])
    tr_t[..., 0] = tr_t[..., 0] + rot_val_t

    bl_t = bl[exceed_t].reshape(1, 1, -1, 2)
    bl_t[..., 1] = torch.ones_like(bl_t[..., 0]) - overhang_t
    bl_t[..., 0] = bl_t[..., 0] + rot_val_t
    br_t = br[exceed_t].reshape(1, 1, -1, 2)
    br_t[..., 1] = torch.ones_like(br_t[..., 0]) - overhang_t
    br_t[..., 0] = br_t[..., 0] + rot_val_t

    rect_t = integrate_area_wrap_lr(bl_t, br_t, tl_t, tr_t, size[exceed_t.reshape(-1)],
                                    cum_mat, interp_mode=interp_mode, align_corners=align_corners)
    bg_vals[exceed_t.reshape(-1)] += rect_t

    exceed_b = bl[..., 1] < -1
    # set bot to be the bot of the pano
    # set top to be bot of pano + overhang
    tl_b = tl[exceed_b].reshape(1, 1, -1, 2)
    rot_val_b = torch.where(tl_b[..., 0] > 0, -1, 1)

    bl_b = bl[exceed_b].reshape(1, 1, -1, 2)
    bl_b[..., 1] = -torch.ones_like(bl_b[..., 0])
    bl_b[..., 0] = bl_b[..., 0] + rot_val_b
    br_b = br[exceed_b].reshape(1, 1, -1, 2)
    br_b[..., 1] = -torch.ones_like(br_b[..., 0])
    br_b[..., 0] = br_b[..., 0] + rot_val_b

    overhang_b = (-1-bl[exceed_b][..., 1]).clip(max=0.5, min=0)
    tl_b[..., 1] = -torch.ones_like(tl_b[..., 0]) + overhang_b
    tl_b[..., 0] = tl_b[..., 0] + rot_val_b
    tr_b = tr[exceed_b].reshape(1, 1, -1, 2)
    tr_b[..., 1] = -torch.ones_like(tr_b[..., 0]) + overhang_b
    tr_b[..., 0] = tr_b[..., 0] + rot_val_b

    rect_b = integrate_area_wrap_lr(bl_b, br_b, tl_b, tr_b, size[exceed_b.reshape(-1)],
                                    cum_mat, interp_mode=interp_mode, align_corners=align_corners)
    bg_vals[exceed_b.reshape(-1)] += rect_b

    return bg_vals

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
        self.align_corners = True
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

        sh_A = torch.tensor([3.141593, *([2.094395]*3), *([0.785398]*5), *([0]*7), *([-0.1309]*9)])
        sh_A = torch.tensor(sum([[sh.Al2(l)]*(2*l+1) for l in range(16)], []))
        self.register_buffer('sh_A', sh_A)

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
        w, h = self.bg_mat.shape[-1], self.bg_mat.shape[-2]
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
    def calc_envmap_psnr(self, gt_im, fH=500):
        fW = 2*fH
        # gt im should be numpy
        gH, gW = gt_im.shape[:2]
        gt_im = gt_im[:, ::-1] # flip
        gt_im = np.concatenate([gt_im[:, gW//2:], gt_im[:, :gW//2]], axis=1)
        # import matplotlib.pyplot as plt
        H, W = self.hw()
        gt_im = torch.as_tensor(gt_im)
        gt_im = F.interpolate(gt_im.permute(2, 0, 1).unsqueeze(0), (fH, fW)).squeeze(0).permute(1, 2, 0)
        # plt.imshow(gt_im)
        # plt.show()
        pred_im = self.activation_fn(self.bg_mat[0]).permute(1, 2, 0).detach().cpu()
        pred_im = F.interpolate(pred_im.permute(2, 0, 1).unsqueeze(0), (fH, fW)).squeeze(0).permute(1, 2, 0)

        Y = gt_im.reshape(-1, 3).numpy()
        X = pred_im.reshape(-1, 3).numpy()
        model = linear_model.LinearRegression()
        model.fit(X, Y)
        pred_Y = model.predict(X)
        err_im = (pred_Y - Y)**2
        # plt.imshow(err_im)
        # plt.show()
        psnr = -10.0 * np.log(err_im.mean()) / np.log(10.0)
        return psnr


    def get_spherical_harmonics(self, G, mipval=-5):
        device = self.get_device()
        _theta = torch.linspace(0, np.pi, G//2, device=device)
        _phi = torch.linspace(0, 2*np.pi, G, device=device)
        theta, phi = torch.meshgrid(_theta, _phi, indexing='ij')
        sh_samples = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=-1).reshape(-1, 3)
        # compute 
        SB = sh_samples.shape[0]
        samp_mipval = mipval*torch.ones(SB, 1, device=device)
        bg = self(sh_samples, samp_mipval)
        evaled = sh.eval_sh_bases(9, sh_samples)
        # evaled = sh.sh_basis([0, 1, 2, 4, 8, 16], sh_samples)
        # evaled: (N, 9)
        # bg: (N, 3)
        # coeffs: (1, 3, 9)
        # integral:
        coeffs = 2*np.pi**2 *(bg.reshape(SB, 1, 3) * evaled.reshape(SB, -1, 1) * torch.sin(theta.reshape(SB, 1, 1))).mean(dim=0)
        # cols = (coeffs.reshape(1, -1, 3) * evaled.reshape(SB, -1, 1)).sum(dim=1)
        # ic(cols, bg)
        conv_coeffs = self.sh_A.reshape(-1, 1)[:coeffs.shape[0]] * coeffs
        # cols = (conv_coeffs.reshape(1, -1, 3) * evaled.reshape(SB, -1, 1)).sum(dim=1)
        return coeffs, conv_coeffs / np.pi

    @torch.no_grad()
    def save(self, path, prefix='', tonemap=None):
        im = self.activation_fn(self.bg_mat)
        if tonemap is not None:
            im = tonemap(im)
        im = im.clamp(0, 1)
        im = (255*im).short().permute(0, 2, 3, 1).squeeze(0)
        im = im.cpu().numpy()
        # im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
        imageio.imwrite(str(path / f'{prefix}pano.png'), im)

    def sa2mip(self, u, saSample, eps=torch.finfo(torch.float32).eps):
        h, w = self.hw()
        saSample = saSample.reshape(-1)
        # TODO calculate distortion of cube map for saTexel

        cos = (1-u[:, 2]**2).clip(min=eps).sqrt()
        d = h * w / (2 * math.pi**2 * cos).clip(min=eps)
        area = ((d/2).log() + saSample).exp()
        base_rect_area = 1/cos.clip(min=eps)
        h = (area.clip(min=eps).sqrt() * cos).clip(min=eps)#.clip(max=area)
        w = area / h
        # ic(h, w, area, cos)
        # b*(x * y) = a
        # miplevel_w = (((d/2).log() + saSample) / math.log(2)) / 2
        # miplevel_h = (((d/2).log() + saSample) / math.log(2)) / 2
        miplevel_w = (w.log() / math.log(2))
        miplevel_h = (h.log() / math.log(2))
        
        miplevel_w = miplevel_w + self.mipbias + self.mipnoise * torch.rand_like(saSample)
        miplevel_h = miplevel_h + self.mipbias + self.mipnoise * torch.rand_like(saSample)
        return miplevel_w.clip(0, 7), miplevel_h.clip(0, 7)

    # def sa2mip(self, u, saSample, eps=torch.finfo(torch.float32).eps):
    #     h, w = self.hw()
    #     # TODO calculate distortion of cube map for saTexel
    #     # distortion = 4 * math.pi / 6
    #     # distortion_w = 1/(1-u[:, 2]**2).clip(min=eps)
    #     distortion_w = (1-u[:, 2]**2).clip(min=eps).sqrt()
    #     distortion_h = torch.ones_like(distortion_w)
    #     saSample = saSample.reshape(-1)
    #
    #     saTexel_w = distortion_w / h / w
    #     saTexel_h = distortion_h / h / w
    #     # saTexel is the ratio to the solid angle subtended by one pixel of the 0th mipmap level
    #     # num_pixels = self.bg_mat.numel() // 3
    #     # saTexel = distortion / num_pixels
    #     miplevel_w = ((saSample - torch.log(saTexel_w.clip(min=eps))) / math.log(2))/2 + self.mipbias + self.mipnoise * torch.rand_like(saSample)
    #     miplevel_h = ((saSample - torch.log(saTexel_h.clip(min=eps))) / math.log(2))/2 + self.mipbias + self.mipnoise * torch.rand_like(saSample)
    #     
    #     return miplevel_w.clip(0), miplevel_h.clip(0)

    def tv_loss(self):
        loss = 0
        # img = self.activation_fn(imgs[0, i])
        img = self.bg_mat[0]
        # img.shape: h, w, 3
        tv_h = (img[1:, :-1] - img[:-1, :-1]).abs()
        tv_w = (img[:-1, 1:] - img[:-1, :-1]).abs()
        loss = (tv_h + tv_w+1e-8).mean()
        return loss

    def forward(self, viewdirs, saSample, max_level=None):
        # Compute width and height of sample
        device = viewdirs.device
        eps = torch.finfo(torch.float32).eps

        miplevel_w, miplevel_h = self.sa2mip(viewdirs, saSample)
        h, w = self.hw()
        sw = 2 ** miplevel_w / h / 2
        sh = 2 ** miplevel_h / h
        # h, w = self.hw()
        # sw = 2 ** saSample / h / 2
        # sh = 2 ** saSample / h
        # ic(sh.min(), sh.max())
        # sw = 0.20*torch.ones_like(sw)
        # sh = 0.20*torch.ones_like(sh)
        offset = torch.stack([sw, sh], dim=-1).reshape(1, 1, -1, 2)

        activated = self.activation_fn(self.bg_mat)
        cum_mat = torch.cumsum(torch.cumsum(activated, dim=2), dim=3)
        # cum_mat = torch.cumsum(self.bg_mat, dim=2)
        # cum_mat = torch.cumsum(self.bg_mat, dim=3)
        # cum_mat = self.bg_mat
        # ic(cum_mat)
        size = (offset / 2 * torch.tensor([w, h], device=device).reshape(1, 1, 1, 2)).prod(dim=-1).reshape(-1, 1)


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
        bg_vals = integrate_area_wrap(bl, br, tl, tr, size, cum_mat,
                                      interp_mode=self.interp_mode, align_corners=self.align_corners)

        # handle top and bottom
        cutoff = 1 - 2 / h * 3
        top_row = activated[..., 0, :].mean(dim=-1)
        bot_row = activated[..., -1, :].mean(dim=-1)
        bg_vals[coords[:, 1] > cutoff] = bot_row
        bg_vals[coords[:, 1] < -cutoff] = top_row
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
    # bg_module.bg_mat.data += -5
    # bg_module.bg_mat.data[0, 0, new_w//2, new_w] = 1
    # bg_module.bg_mat.data[0, 0, new_w//4+1, new_w] = 1
    # bg_module.bg_mat.data[0, 0, new_w//8+1, new_w] = 1
    # bg_module.bg_mat.data[0, 0, new_w//2-1, new_w//2] = 1

    bg_module = bg_module.to(device)
    res = bg_res
    ele_grid, azi_grid = torch.meshgrid(
        torch.linspace(-math.pi/2, math.pi/2, res, dtype=torch.float32),
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
    mipvals, _ = bg_module.sa2mip(ang_vecs, sa_sample)
    mipvals = mipvals.reshape(res, 2*res)
    ic(mipvals.shape)
    blur_img = vals.reshape(res, 2*res, 3)#.abs()
    imageio.imwrite("bg_blur.exr", blur_img.detach().cpu().numpy())
    imageio.imwrite("bg_blur_abs.exr", blur_img.abs().detach().cpu().numpy())
    imageio.imwrite("bg_noblur.exr", bg_module.bg_mat.exp().squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
    imageio.imwrite("mipvals.exr", mipvals.detach().cpu().numpy())

    # ang_vecs = torch.tensor([1, 0, 0]).reshape(1, 3).to(device)
    # sa_sample = -5*torch.ones((ang_vecs.shape[0], 1), device=device)
    # vals = bg_module(ang_vecs, sa_sample)
    # ic(vals)

