import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .render_modules import positional_encoding, str2fn
from icecream import ic
import matplotlib.pyplot as plt
from . import safemath
from mutils import normalize

from modules import util

import plotly.express as px
import plotly.graph_objects as go

def schlick(f0, n, l):
    return f0 + (1-f0)*(1-(n*l).sum(dim=-1, keepdim=True).clip(min=1e-20))**5

def ggx_dist(NdotH, roughness):
    # takes the cos of the zenith angle between the micro surface and the macro surface
    # and returns the probability of that micro surface existing
    a2 = roughness**2
    # return a2 / np.pi / ((NdotH**2*(a2-1)+1)**2).clip(min=1e-8)
    return ((a2 / (NdotH.clip(min=0, max=1)**2*(a2-1)+1))**2).clip(min=0, max=1)


class PBR(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lr=0

    def forward(self, incoming_light, V, L, N, features, roughness, matprop, mask, ray_mask):
        # V: (B, 3)-viewdirs, the outgoing light direction
        # L: (B, M, 3) incoming light direction. bounce_rays
        # N: (B, 3) outward normal
        # matprop: dictionary of attributes
        # mask: mask for matprop
        half = normalize(L + V.reshape(-1, 1, 3))

        LdotN = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        VdotN = (V.reshape(-1, 1, 3) * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        cos_half = (half * N).sum(dim=-1, keepdim=True)

        # compute the BRDF (bidirectional reflectance distribution function)
        # k_d = ratio_diffuse[bounce_mask].reshape(-1, 1, 1)
        # diffuse vs specular fraction
        # r = matprop['reflectivity'][mask]
        # 0.04 is the approximate value of F0 for non-metallic surfaces
        # f0 = (1-r)*0.04 + r*matprop['tint'][mask]

        f0 = matprop['f0'][mask].reshape(-1, 1, 3)
        # f0 = matprop['tint'][mask].reshape(-1, 1, 3)

        k_s = schlick(f0, N.double(), half.double()).float()

        # diffuse vs specular intensity
        # f_d = matprop['reflectivity'].reshape(-1, 1, 1)/np.pi/M
        k_d = 1-k_s

        # alph = 0*matprop['roughness'][mask].reshape(-1, 1, 1) + 0.1
        alph = roughness.reshape(-1, 1, 1)
        a2 = alph**2
        # k = alph**2 / 2 # ibl
        # a2 = alph**2
        D_ggx = ggx_dist(cos_half, alph)
        k = (alph+1)**2 / 8 # direct lighting
        G_schlick_smith = 1 / ((VdotN*(1-k)+k)*(LdotN*(1-k)+k)).clip(min=1e-8)
        
        # f_s = D_ggx.clip(min=0, max=1) * G_schlick_smith.clip(min=0, max=1) / (4 * LdotN * VdotN).clip(min=1e-8)
        # f_s = D_ggx / (4 * LdotN * VdotN).clip(min=1e-8)

        f_s = D_ggx * G_schlick_smith / 4
        # f_s = D_ggx / (4 * LdotN * VdotN).clip(min=1e-8)


        # the diffuse light is covered by other components of the rendering equation
        # ic(k_d.mean(dim=1).mean(dim=0), k_s.mean(dim=1).mean(dim=0))
        # brdf = k_d*f_d + k_s*f_s
        albedo = matprop['albedo'][mask].reshape(-1, 1, 3)
        brdf = k_d*albedo + k_s*f_s
        brdf *= ray_mask
        # normalize probabilities so they sum to 1. the rgb dims represent the spectra in equal parts.
        brdf = brdf / brdf.sum(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        # brdf = k_s * f_s

        # cos_refl = (noise_rays * refdirs[full_bounce_mask].reshape(-1, 1, 3)).sum(dim=-1, keepdim=True).clip(min=0)
        # cos_refl = (bounce_rays[..., 3:6] * refdirs[full_bounce_mask].reshape(-1, 1, 3)).sum(dim=-1, keepdim=True).clip(min=0)
        # ref_rough = roughness[bounce_mask].reshape(-1, 1)
        # phong shading?
        # ref_weight = (1-ref_rough) * cos_refl + ref_rough * LdotN
        ref_weight = brdf * LdotN
        spec_color = (incoming_light * ref_weight).sum(dim=1)
        return spec_color


class MLPBRDF(torch.nn.Module):
    def __init__(self, in_channels, h_encoder=None, d_encoder=None, v_encoder=None, n_encoder=None, l_encoder=None, feape=6, dotpe=0,
                 activation='sigmoid', mul_LdotN=True, bias=0, lr=1e-4, shift=0, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.dotpe = dotpe
        self.bias = bias
        self.in_mlpC = 2*feape*in_channels + in_channels
        if dotpe >= 0:
            self.in_mlpC += 6 + 2*dotpe*6
        self.v_encoder = v_encoder
        self.n_encoder = n_encoder
        self.l_encoder = l_encoder
        self.h_encoder = h_encoder
        self.d_encoder = d_encoder
        self.mul_LdotN = mul_LdotN
        self.lr = lr
        self.activation_name = activation
        if h_encoder is not None:
            self.in_mlpC += self.h_encoder.dim() + 3
        if d_encoder is not None:
            self.in_mlpC += self.d_encoder.dim() + 3
        if v_encoder is not None:
            self.in_mlpC += self.v_encoder.dim() + 3
        if n_encoder is not None:
            self.in_mlpC += self.n_encoder.dim() + 3
        if l_encoder is not None:
            self.in_mlpC += self.l_encoder.dim() + 3

        self.feape = feape
        self.mlp = util.create_mlp(self.in_mlpC, 4, **kwargs)
        # self.activation = str2fn(activation)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # if m.weight.shape[0] <= 4:
            #     torch.nn.init.constant_(m.weight, np.sqrt(2) / m.weight.shape[1])
            # else:
            #     torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def activation(self, x):
        if self.activation_name == 'sigexp':
            col = torch.sigmoid(x[..., :3])
            brightness = torch.exp(x[..., 3:4].clip(min=-10, max=10)-1)
            return col * brightness
        else:
            return str2fn(self.activation_name)(x[..., :3]+self.bias)
            # raise Exception(f"{self.activation} not implemented in BRDF")
        # return torch.sigmoid(x)
        # return F.softplus(x+1.0)/2

    # def save(self, features, path, res=50):
    #     device = features.device
    #     ele_grid, azi_grid = torch.meshgrid(
    #         torch.linspace(-np.pi/2, np.pi/2, res, dtype=torch.float32),
    #         torch.linspace(-np.pi, np.pi, 2*res, dtype=torch.float32), indexing='ij')
    #     # each col of x ranges from -pi/2 to pi/2
    #     # each row of y ranges from -pi to pi
    #     ang_vecs = torch.stack([
    #         torch.cos(ele_grid) * torch.cos(azi_grid),
    #         torch.cos(ele_grid) * torch.sin(azi_grid),
    #         -torch.sin(ele_grid),
    #     ], dim=-1).to(device)
    #     static_vecs = torch.zeros_like(ang_vecs)
    #     self()
        

    def forward(self, V, L, N, half_vec, diff_vec, efeatures, eroughness):
        # V: (n, 3)-viewdirs, the outgoing light direction
        # L: (n, m, 3) incoming light direction. bounce_rays
        # N: (n, 1, 3) outward normal
        # features: (B, D)
        # matprop: dictionary of attributes
        # mask: mask for matprop

        half = normalize(L + V)

        LdotN = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        if self.dotpe >= 0:

            VdotN = (V * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
            NdotH = ((half * N).sum(dim=-1, keepdim=True)+1e-3).clip(min=1e-20, max=1)
            indata = [LdotN, torch.sqrt((1-LdotN**2).clip(min=1e-8, max=1)),
                      VdotN, torch.sqrt((1-LdotN**2).clip(min=1e-8, max=1)),
                      NdotH, torch.sqrt((1-NdotH**2).clip(min=1e-8, max=1))]
            indata = [d.reshape(-1, 1) for d in indata]
            if self.dotpe > 0:
                dotvals = torch.cat(indata, dim=-1)
                indata += [safemath.integrated_pos_enc((dotvals * torch.pi, 0.20*torch.ones_like(dotvals)), 0, self.dotpe)]
        else:
            indata = []
        # indata = [safemath.arccos(LdotN.reshape(-1, 1)), safemath.arccos(VdotN.reshape(-1, 1)), safemath.arccos(NdotH.reshape(-1, 1))]
        indata += [efeatures]

        L = L.reshape(-1, 3)
        B = V.shape[0]
        if self.h_encoder is not None:
            indata += [self.h_encoder(half_vec, eroughness).reshape(B, -1), half_vec]
        if self.d_encoder is not None:
            indata += [self.d_encoder(diff_vec, eroughness).reshape(B, -1), diff_vec]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.v_encoder is not None:
            indata += [self.v_encoder(V, eroughness).reshape(B, -1), V]
        if self.n_encoder is not None:
            indata += [self.n_encoder(N, eroughness).reshape(B, -1), N]
        if self.l_encoder is not None:
            indata += [self.l_encoder(L, eroughness).reshape(B, -1), L]

        # ic("H")
        # for d in indata:
        #     ic(d.shape)

        mlp_in = torch.cat(indata, dim=-1)
        raw_mlp_out = self.mlp(mlp_in)
        mlp_out = self.activation(raw_mlp_out[..., :4])
        # k_s = torch.sigmoid(raw_mlp_out[..., 4:5]-1).clip(min=0.01)
        # ic(f0.mean())
        ref_weight = mlp_out[..., :3]

        if self.mul_LdotN:
            weight = ref_weight * LdotN.abs().detach()
        else:
            weight = ref_weight

        # plot it
        # splat_weight = torch.zeros((*ray_mask.shape, 3), dtype=weight.dtype, device=weight.device)
        # splat_L = torch.zeros((*ray_mask.shape, 3), dtype=weight.dtype, device=weight.device)
        # splat_weight[ray_mask] = weight
        # splat_L[ray_mask] = L
        # ind = ray_mask.sum(dim=1).argmax()
        # w = splat_weight[ind].detach().cpu()
        # ls = splat_L[ind].detach().cpu()
        # px.scatter_3d(x=ls[:, 0], y=ls[:, 1], z=ls[:, 2], color=w.mean(dim=-1)).show()
        # assert(False)
        return weight
