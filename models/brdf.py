import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .render_modules import positional_encoding
from icecream import ic

def schlick(f0, n, l):
    return f0 + (1-f0)*(1-(n*l).sum(dim=-1, keepdim=True).clip(min=1e-20))**5

class SimplePBR(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, V, L, N, features, matprop, mask):
        cos_lamb = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        ref_weight = cos_lamb / cos_lamb.sum(dim=1, keepdim=True)
        return ref_weight

class PBR(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, V, L, N, features, matprop, mask):
        # V: -viewdirs, the outgoing light direction
        # L: incoming light direction. bounce_rays
        # N: outward normal
        # matprop: dictionary of attributes
        # mask: mask for matprop
        half = L + V.reshape(-1, 1, 3)
        half = half / (torch.linalg.norm(half, dim=-1, keepdim=True)+1e-8)

        cos_lamb = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        cos_view = (V.reshape(-1, 1, 3) * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        cos_half = (half * N).sum(dim=-1, keepdim=True).clip(min=1e-8)

        # compute the BRDF (bidirectional reflectance distribution function)
        # k_d = ratio_diffuse[bounce_mask].reshape(-1, 1, 1)
        # diffuse vs specular fraction
        r = matprop['reflectivity'][mask]
        f0 = (1-r)*0.04 + r*matprop['tint'][mask]
        k_s = schlick(f0.reshape(-1, 1, 3), N.double(), half.double()).float()
        k_d = 1-k_s

        # diffuse vs specular intensity
        f_d = 1/np.pi

        # alph = 0*matprop['roughness'][mask].reshape(-1, 1, 1) + 0.1
        alph = matprop['roughness'][mask].reshape(-1, 1, 1)
        k = (alph+1)**2 / 8
        a2 = alph**4
        # k = alph / 2
        # a2 = alph**2
        D_ggx = (a2 / (np.pi * (cos_half**2*(a2-1)+1)**2).clip(min=1e-10))
        G_schlick_smith = (cos_lamb * cos_view / ((cos_view*(1-k)+k)*(cos_lamb*(1-k)+k)).clip(min=1e-8))
        
        # f_s = D_ggx.clip(min=0, max=1) * G_schlick_smith.clip(min=0, max=1) / (4 * cos_lamb * cos_view).clip(min=1e-8)
        f_s = D_ggx * G_schlick_smith / (4 * cos_lamb * cos_view).clip(min=1e-8)
        # brdf = k_d*f_d + k_s*f_s
        # the diffuse light is covered by other components of the rendering equation
        f_s = f_s / f_s.sum(dim=1, keepdim=True)
        brdf = k_s*f_s
        # ic(k_s.mean(dim=1).mean(dim=0))

        # cos_refl = (noise_rays * refdirs[full_bounce_mask].reshape(-1, 1, 3)).sum(dim=-1, keepdim=True).clip(min=0)
        # cos_refl = (bounce_rays[..., 3:6] * refdirs[full_bounce_mask].reshape(-1, 1, 3)).sum(dim=-1, keepdim=True).clip(min=0)
        # ref_rough = roughness[bounce_mask].reshape(-1, 1)
        # phong shading?
        # ref_weight = (1-ref_rough) * cos_refl + ref_rough * cos_lamb
        # ref_weight = ref_weight / (ref_weight.sum(dim=-1, keepdim=True)+1e-8)

        # tinted_ref_rgb = (ref_weight * incoming_light).mean(dim=1).float()

        # aden = (-bn*l[bounce_mask].reshape(-1, 1, 3)).sum(dim=-1, keepdim=True).clip(min=0)
        # bden = (bn*bounce_rays[..., 3:6]).sum(dim=-1, keepdim=True).clip(min=0)
        # denom = 1
        # tinted_ref_rgb = (numer / (denom+1e-8) * incoming_light).mean(dim=1).float()
        # tinted_ref_rgb = (numer / (denom+1e-8)).mean(dim=1).float()
        # ic(k_s[:, 0], f_s[:, 0], D_ggx[:, 0], G_schlick_smith[:, 0]) 

        # tinted_ref_rgb = (brdf * incoming_light * cos_lamb).mean(dim=1).float()
        ref_weight = (brdf * cos_lamb)
        # ic(brdf.min(), brdf.max(), k_s.max(), f_s.max(), D_ggx.max(), G_schlick_smith.max(), ref_weight.max(), ref_weight.sum(dim=1).mean(dim=0))
        # ref_weight = ref_weight / (ref_weight.sum(dim=1, keepdim=True).max(dim=2, keepdim=True).values+1e-8)
        # ic(matprop['f0'].shape, brdf.shape, k_s.shape, f_s.shape, ref_weight.shape)
        # ic(ref_weight.sum(dim=1).mean(dim=0))
        return ref_weight

class MLPBRDF(torch.nn.Module):
    def __init__(self, in_channels, v_encoder=None, n_encoder=None, l_encoder=None, feape=6, featureC=128, num_layers=2):
        super().__init__()

        self.in_channels = in_channels
        self.in_mlpC = 2*feape*in_channels + in_channels
        self.v_encoder = v_encoder
        self.n_encoder = n_encoder
        self.l_encoder = l_encoder
        if v_encoder is not None:
            self.in_mlpC += self.v_encoder.dim() + 3
        if n_encoder is not None:
            self.in_mlpC += self.n_encoder.dim() + 3
        if l_encoder is not None:
            self.in_mlpC += self.l_encoder.dim() + 3

        self.feape = feape
        if num_layers > 0:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.in_mlpC, featureC),
                # torch.nn.ReLU(inplace=True),
                # torch.nn.Linear(featureC, featureC),
                # torch.nn.BatchNorm1d(featureC),
                *sum([[
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(featureC, featureC),
                        # torch.nn.BatchNorm1d(featureC)
                    ] for _ in range(num_layers)], []),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(featureC, 3),
            )
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
            # self.mlp.apply(self.init_weights)
        else:
            self.mlp = torch.nn.Identity()

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def forward(self, V, L, N, features, matprop, mask):
        # V: (n, 3)-viewdirs, the outgoing light direction
        # L: (n, m, 3) incoming light direction. bounce_rays
        # N: (n, 1, 3) outward normal
        # features: (B, D)
        # matprop: dictionary of attributes
        # mask: mask for matprop
        roughness = matprop['roughness'][mask]
        D = features.shape[-1]
        n, m, _ = L.shape
        features = features.reshape(n, 1, D).expand(n, m, D).reshape(-1, D)
        indata = [features]

        V = V.reshape(n, 1, 3).expand(L.shape).reshape(-1, 3)
        N = N.expand(n, m, 3).reshape(-1, 3)
        L = L.reshape(-1, 3)
        B = V.shape[0]

        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.v_encoder is not None:
            indata += [self.v_encoder(V, roughness).reshape(B, -1), V]
        if self.n_encoder is not None:
            indata += [self.n_encoder(N, roughness).reshape(B, -1), N]
        if self.l_encoder is not None:
            indata += [self.l_encoder(L, roughness).reshape(B, -1), L]

        mlp_in = torch.cat(indata, dim=-1)
        ref_weight = self.mlp(mlp_in)
        ref_weight = torch.softmax(ref_weight.reshape(n, m, -1), dim=1)
        return ref_weight

