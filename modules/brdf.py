import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .render_modules import positional_encoding, str2fn
from icecream import ic
import matplotlib.pyplot as plt
from . import safemath
from mutils import normalize, signed_clip

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

def aniso_smith_masking_gtr2(v_local, ax, ay, eps=torch.finfo(torch.float32).eps):
    v2 = v_local * v_local
    Lambda = (-1 + (1+(v2[..., 0] * ax*ax + v2[..., 1] * ay*ay) / signed_clip(v2[..., 2])).clip(min=eps).sqrt()) / 2
    return 1 / (1+Lambda)

class Specular(torch.nn.Module):
    def __init__(self, in_channels, lr, bias, **kwargs):
        super().__init__()
        self.lr=lr
        self.in_channels = in_channels
        self.in_mlpC = in_channels
        self.bias = bias
        self.C0_mlp = util.create_mlp(self.in_mlpC, 3, **kwargs)

    def calibrate(self, efeatures, bg_brightness):
        pass

    def forward(self, V, L, N, local_v, half_vec, diff_vec, efeatures, ax, ay):
        LdotH = (diff_vec * half_vec).sum(dim=-1, keepdim=True)
        VdotH = (local_v * half_vec).sum(dim=-1, keepdim=True)

        C_0 = torch.sigmoid(self.C0_mlp(efeatures)+self.bias)
        Fm = C_0 + (1-C_0) * VdotH**5
        Gm = aniso_smith_masking_gtr2(diff_vec, ax, ay) * aniso_smith_masking_gtr2(local_v, ax, ay)
        # spec = (Fm * Gm.reshape(-1, 1)) / (4 * LdotH.abs()).clip(min=1e-8)
        # spec = (Fm) / 4
        spec = (Fm * Gm.reshape(-1, 1)) / 4
        # ic(C_0, spec.max(dim=0), spec)
        return spec#.clip(max=1)


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

    def calibrate(self, efeatures, bg_brightness):
        N = efeatures.shape[0]
        device = efeatures.device
        def rand_vecs():
            v = normalize(2*torch.rand((N, 3), device=device) - 1)
            return v
        weight = self(rand_vecs(), rand_vecs(), rand_vecs(), rand_vecs(), rand_vecs(), rand_vecs(), efeatures, torch.rand((N), device=device), torch.rand((N), device=device))
        # ic(self(rand_vecs(), rand_vecs(), rand_vecs(), rand_vecs(), rand_vecs(), efeatures, torch.rand((N), device=device)).mean())
        target_val = 0.25 / bg_brightness.item()
        ic(bg_brightness, target_val)
        self.bias += math.log(target_val / (1-target_val)) - (weight / (1-weight)).log().mean().detach().item()
        # ic(self.bias, -(weight / (1-weight)).log().mean().detach().item())
        # ic(self(rand_vecs(), rand_vecs(), rand_vecs(), rand_vecs(), rand_vecs(), efeatures, torch.rand((N), device=device)).mean())


    def forward(self, V, L, N, local_v, half_vec, diff_vec, efeatures, eax, eay):
        # V: (n, 3)-viewdirs, the outgoing light direction
        # L: (n, m, 3) incoming light direction. bounce_rays
        # N: (n, 1, 3) outward normal
        # features: (B, D)
        # matprop: dictionary of attributes
        # mask: mask for matprop


        LdotN = (L * N).sum(dim=-1, keepdim=True)
        LdotH = (diff_vec * half_vec).sum(dim=-1, keepdim=True)
        if self.dotpe >= 0:

            VdotN = (V * N).sum(dim=-1, keepdim=True)
            NdotH = half_vec[..., 2]
            # indata = [LdotN, torch.sqrt((1-LdotN**2).clip(min=1e-8, max=1)),
            #           VdotN, torch.sqrt((1-LdotN**2).clip(min=1e-8, max=1)),
            #           NdotH, torch.sqrt((1-NdotH**2).clip(min=1e-8, max=1))]
            indata = [LdotH, torch.sqrt((1-LdotN**2).clip(min=1e-8, max=1)),
                      # LdotN, torch.sqrt((1-LdotN**2).clip(min=1e-8, max=1)),
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
            indata += [self.h_encoder(half_vec, eax).reshape(B, -1), half_vec]
        if self.d_encoder is not None:
            indata += [self.d_encoder(diff_vec, eax).reshape(B, -1), diff_vec]
        if self.feape > 0:
            indata += [positional_encoding(efeatures, self.feape)]
        if self.v_encoder is not None:
            indata += [self.v_encoder(V, eax).reshape(B, -1), V]
        if self.n_encoder is not None:
            indata += [self.n_encoder(N, eax).reshape(B, -1), N]
        if self.l_encoder is not None:
            indata += [self.l_encoder(L, eax).reshape(B, -1), L]

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
