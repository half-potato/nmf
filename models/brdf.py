import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .render_modules import positional_encoding, str2fn
from icecream import ic
import matplotlib.pyplot as plt
from . import safemath
from modules import row_mask_sum

import plotly.express as px
import plotly.graph_objects as go

def schlick(f0, n, l):
    return f0 + (1-f0)*(1-(n*l).sum(dim=-1, keepdim=True).clip(min=1e-20))**5

def normalize(x):
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True)+1e-8)

def ggx_dist(NdotH, roughness):
    # takes the cos of the zenith angle between the micro surface and the macro surface
    # and returns the probability of that micro surface existing
    a2 = roughness**2
    # return a2 / np.pi / ((NdotH**2*(a2-1)+1)**2).clip(min=1e-8)
    return ((a2 / (NdotH.clip(min=0, max=1)**2*(a2-1)+1))**2).clip(min=0, max=1)

class NormalSampler:
    def __init__(self) -> None:
        pass

    def roughness2noisestd(self, roughness):
        # return roughness
        # slope = 6
        # TODO REMOVE
        return roughness

    def sample(self, num_samples, refdirs, viewdir, normal, roughness):
        device = normal.device
        B = normal.shape[0]
        ray_noise = torch.normal(0, 1, (B, num_samples, 3), device=device)
        diffuse_noise = ray_noise / (torch.linalg.norm(ray_noise, dim=-1, keepdim=True)+1e-8)
        diffuse_noise = self.roughness2noisestd(roughness.reshape(-1, 1, 1)) * diffuse_noise
        outward = normal.reshape(-1, 1, 3)

        # this formulation uses ratio diffuse and ratio reflected to do importance sampling
        # diffuse_noise = ratio_diffuse[bounce_mask].reshape(-1, 1, 1) * (outward+diffuse_noise)
        # diffuse_noise[:, 0] = 0
        # noise_rays = ratio_reflected[bounce_mask].reshape(-1, 1, 1) * (brefdirs + reflect_noise) + diffuse_noise

        # this formulation uses roughness to do importance sampling
        # diffuse_noise = outward+diffuse_noise
        diffuse_noise[:, 0] = 0
        # noise_rays = (1-roughness[bounce_mask].reshape(-1, 1, 1)) * brefdirs + roughness[bounce_mask].reshape(-1, 1, 1) * diffuse_noise
        noise_rays = refdirs + diffuse_noise

        # noise_rays = ratio_reflected[bounce_mask].reshape(-1, 1, 1) * (bounce_rays[..., 3:6] + reflect_noise)
        noise_rays = noise_rays / (torch.linalg.norm(noise_rays, dim=-1, keepdim=True)+1e-8)
        # project into 
        return noise_rays

class GGXSampler(torch.nn.Module):
    def __init__(self, num_samples) -> None:
        super().__init__()
        self.sampler = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        self.num_samples = num_samples
        angs = self.sampler.draw(num_samples)
        self.register_buffer('angs', angs)
        # plt.scatter(self.angs[:, 0], self.angs[:, 1])
        # plt.show()

    def draw(self, B, num_samples):
        # self.angs = self.sampler.draw(self.num_samples)
        angs = self.angs.reshape(1, self.num_samples, 2)[:, :num_samples, :].expand(B, num_samples, 2)
        # self.sampler = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        # add random offset
        offset = torch.rand(B, 1, 2, device=angs.device)*0.25
        angs = (angs + offset) % 1.0
        return angs

    def update(self, *args, **kwargs):
        pass

    def sample(self, num_samples, refdirs, viewdir, normal, roughness, ray_mask):
        # viewdir: (B, 3)
        # normal: (B, 3)
        # roughness: B
        device = normal.device
        B = normal.shape[0]

        # establish basis for BRDF
        z_up = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, 3).expand(B, 3)
        x_up = torch.tensor([1.0, 0.0, 0.0], device=device).reshape(1, 3).expand(B, 3)
        up = torch.where(normal[:, 2:3] < 0.9, z_up, x_up)
        tangent = normalize(torch.linalg.cross(up, normal))
        bitangent = normalize(torch.linalg.cross(normal, tangent))
        # B, 3, 3
        row_world_basis = torch.stack([tangent, bitangent, normal], dim=1).reshape(B, 3, 3)


        # GGXVNDF
        # V_l = torch.matmul(torch.inverse(row_world_basis.permute(0, 2, 1)), viewdir.unsqueeze(-1)).squeeze(-1)
        # ic((normal*viewdir).sum(dim=-1).min(), (normal*viewdir).sum(dim=-1).max())
        # ic(1, V_l.min(dim=0), V_l.max(dim=0))
        V_l = torch.matmul(row_world_basis, viewdir.unsqueeze(-1)).squeeze(-1)
        # ic(2, V_l.min(dim=0), V_l.max(dim=0))
        V_stretch = normalize(torch.stack([roughness*V_l[..., 0], roughness*V_l[..., 1], V_l[..., 2]], dim=-1)).unsqueeze(1)
        T1 = torch.where(V_stretch[..., 2:3] < 0.999, normalize(torch.linalg.cross(V_stretch, z_up.unsqueeze(1), dim=-1)), x_up.unsqueeze(1))
        T2 = normalize(torch.linalg.cross(T1, V_stretch, dim=-1))
        z = V_stretch[..., 2].reshape(-1, 1)
        a = (1 / (1+z.detach()).clip(min=1e-5)).clip(max=1e4)
        angs = self.draw(B, num_samples).to(device)

        # here is where things get really large
        u1 = angs[..., 0]
        u2 = angs[..., 1]

        # stretch and mask stuff to reduce memory
        a_mask = a.expand(u1.shape)[ray_mask]
        r_mask = roughness.reshape(-1, 1).expand(u1.shape)[ray_mask]
        z_mask = z.expand(u1.shape)[ray_mask]
        u1_mask = u1[ray_mask]
        u2_mask = u2[ray_mask]
        T1_mask = T1.expand(-1, num_samples, 3)[ray_mask]
        T2_mask = T2.expand(-1, num_samples, 3)[ray_mask]
        V_stretch_mask = V_stretch.expand(-1, num_samples, 3)[ray_mask]
        row_world_basis_mask = row_world_basis.permute(0, 2, 1).reshape(B, 1, 3, 3).expand(B, num_samples, 3, 3)[ray_mask]

        r = torch.sqrt(u1_mask)
        phi = torch.where(u2_mask < a_mask, u2_mask/a_mask*math.pi, (u2_mask-a_mask)/(1-a_mask)*math.pi + math.pi)
        P1 = (r*safemath.safe_cos(phi)).unsqueeze(-1)
        P2 = (r*safemath.safe_sin(phi)*torch.where(u2_mask < a_mask, torch.tensor(1.0, device=device), z_mask)).unsqueeze(-1)
        # ic((1-a).min(), a.min(), a.max(), phi.min(), phi.max(), (1-a).max())
        N_stretch = P1*T1_mask + P2*T2_mask + (1 - P1*P1 - P2*P2).clip(min=0).sqrt() * V_stretch_mask
        H_l = normalize(torch.stack([r_mask*N_stretch[..., 0], r_mask*N_stretch[..., 1], N_stretch[..., 2].clip(min=0)], dim=-1))
        H = torch.matmul(row_world_basis_mask, H_l.unsqueeze(-1)).squeeze(-1)
        # H = torch.einsum('bni,bij->bnj', H_l, row_world_basis)

        V = viewdir.unsqueeze(1).expand(-1, num_samples, 3)[ray_mask]
        N = normal.reshape(-1, 1, 3).expand(-1, num_samples, 3)[ray_mask]
        L = (2.0 * (V * H).sum(dim=-1, keepdim=True) * H - V)

        # calculate mipval, which will be used to calculate the mip level
        # half is considered to be the microfacet normal
        # viewdir = incident direction

        # H = normalize(L + viewdir.reshape(-1, 1, 3))

        NdotH = ((H * N).sum(dim=-1)+1e-3).clip(min=1e-8, max=1)
        HdotV = (H * V).sum(dim=-1).abs()
        NdotV = (N * V).sum(dim=-1).abs().clip(min=1e-8, max=1)
        D = ggx_dist(NdotH, r_mask.clip(min=1e-3))
        # ic(NdotH.shape, NdotH, D, D.mean())
        # px.scatter(x=NdotH[0].detach().cpu().flatten(), y=D[0].detach().cpu().flatten()).show()
        # assert(False)
        # ic(NdotH.mean())
        lpdf = torch.log(D.clip(min=1e-5)) + torch.log(HdotV.clip(min=1e-5)) - torch.log(NdotV) - torch.log(r_mask.clip(min=1e-5))
        # pdf = D * HdotV / NdotV / roughness.reshape(-1, 1)
        # pdf = NdotH / 4 / HdotV
        # pdf = D# / NdotH
        mipval = -math.log(num_samples) - lpdf
        # ic(phi.max(), phi.min(), a.min(), a.max(), mipval.min(), mipval.max(), roughness.min(), roughness.max())
        # ic(roughness.grad, lpdf.grad)
        # if phi.grad is not None:
        #     ic(phi.grad.max(), phi.grad.min(), a.grad.min(), a.grad.max(), mipval.grad.min(), mipval.grad.max(), roughness.grad.min(), roughness.grad.max())
        # mipval = 1 / (num_samples * pdf + 1e-6)
        # px.scatter(x=NdotH.detach().cpu().flatten()[:1000], y=mipval.detach().cpu().flatten()[:1000]).show()
        # ic(mipval)
        # assert(False)
        # ic(D.mean(), mipval.mean(), pdf.mean())
        # ic(mipval.mean(dim=1).squeeze(), roughness)
        # ic(mipval[0, 0], D[0, 0], NdotH[0, 0], HdotV[0, 0])

        return L, mipval

class Phong(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lr = 0

    def forward(self, incoming_light, V, L, N, features, matprop, mask, ray_mask):
        B, M, _ = L.shape
        # refdir = L[:, 0:1, :]
        refdir = 2*(N * L).sum(dim=-1, keepdim=True) * N - L
        RdotV = (refdir * V.reshape(-1, 1, 3)).sum(dim=-1, keepdim=True).clip(min=1e-8)#.reshape(-1, 1, 1)
        # ic(RdotV, V, refdir)
        LdotN = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        tint = matprop['tint'][mask].reshape(-1, 1, 3)
        f0 = matprop['f0'][mask].reshape(-1, 1, 3)
        alpha = matprop['reflectivity'][mask].reshape(-1, 1, 1)
        albedo = matprop['albedo'][mask].reshape(-1, 3)
        diffuse = tint * LdotN
        specular = f0 * RdotV**alpha
        output = albedo + ((diffuse + specular) * incoming_light).sum(dim=1) / M

        return output

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
    def __init__(self, in_channels, v_encoder=None, n_encoder=None, l_encoder=None, feape=6, featureC=128, num_layers=2,
                 mul_ggx=False, activation='sigmoid', use_roughness=False, lr=1e-4, detach_roughness=False, shift=0):
        super().__init__()

        self.in_channels = in_channels
        self.use_roughness = use_roughness
        self.detach_roughness = detach_roughness
        self.mul_ggx = mul_ggx
        self.in_mlpC = 2*feape*in_channels + in_channels + (1 if use_roughness else 0) + 6
        # self.in_mlpC = 2*feape*in_channels + (1 if use_roughness else 0) + 6
        self.v_encoder = v_encoder
        self.n_encoder = n_encoder
        self.l_encoder = l_encoder
        self.lr = lr
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
                    ] for _ in range(num_layers-2)], []),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(featureC, 4),
            )
            torch.nn.init.constant_(self.mlp[-1].bias, shift)
            self.mlp.apply(self.init_weights)
        else:
            self.mlp = torch.nn.Identity()
        # self.activation = str2fn(activation)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def activation(self, x):
        # y = torch.tanh(x/10)+1
        # ic(x, y)
        # return y
        col = torch.sigmoid(x[..., :3])
        brightness = torch.exp(x[..., 3:4]).clamp(max=1000)
        return col * brightness
        # return torch.sigmoid(x)
        # return F.softplus(x+1.0)/2

    def forward(self, incoming_light, V, L, N,
            features, roughness, matprop, mask, ray_mask):
        # V: (n, 3)-viewdirs, the outgoing light direction
        # L: (n, m, 3) incoming light direction. bounce_rays
        # N: (n, 1, 3) outward normal
        # features: (B, D)
        # matprop: dictionary of attributes
        # mask: mask for matprop
        D = features.shape[-1]
        device = incoming_light.device
        n, m = ray_mask.shape
        ray_mask = ray_mask.squeeze(-1)

        features = features.reshape(n, 1, D).expand(n, m, D)[ray_mask]
        eroughness = roughness.reshape(-1, 1).expand(n, m)[ray_mask].reshape(-1, 1)
        V_mask = V.reshape(-1, 1, 3).expand(n, m, 3)[ray_mask]
        N_mask = N.reshape(-1, 1, 3).expand(n, m, 3)[ray_mask]
        half = normalize(L + V_mask)

        LdotN = (L * N_mask).sum(dim=-1, keepdim=True).clip(min=1e-8)
        VdotN = (V_mask * N_mask).sum(dim=-1, keepdim=True).clip(min=1e-8)
        NdotH = ((half * N_mask).sum(dim=-1, keepdim=True)+1e-3).clip(min=1e-20, max=1)

        # indata = [LdotN.reshape(-1, 1), VdotN.reshape(-1, 1), NdotH.reshape(-1, 1)]
        indata = [LdotN, torch.sqrt((1-LdotN**2).clip(min=1e-8, max=1)),
                  VdotN, torch.sqrt((1-LdotN**2).clip(min=1e-8, max=1)),
                  NdotH, torch.sqrt((1-NdotH**2).clip(min=1e-8, max=1))]
        # indata = [safemath.arccos(LdotN.reshape(-1, 1)), safemath.arccos(VdotN.reshape(-1, 1)), safemath.arccos(NdotH.reshape(-1, 1))]
        indata = [d.reshape(-1, 1) for d in indata]
        indata += [features]

        # ic(indata)
        # indata = [features]
        if self.detach_roughness:
            eroughness = eroughness.detach()
        if self.use_roughness:
            indata.append(eroughness)

        L = L.reshape(-1, 3)
        B = V.shape[0]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.v_encoder is not None:
            indata += [self.v_encoder(V_mask, eroughness).reshape(B, -1), V_mask]
        if self.n_encoder is not None:
            indata += [self.n_encoder(N_mask, eroughness).reshape(B, -1), N_mask]
        if self.l_encoder is not None:
            indata += [self.l_encoder(L, eroughness).reshape(B, -1), L]

        mlp_in = torch.cat(indata, dim=-1)
        raw_mlp_out = self.mlp(mlp_in)
        mlp_out = self.activation(raw_mlp_out)
        ref_weight = mlp_out[..., :3]

        if self.mul_ggx:
            D = ggx_dist(NdotH, roughness.reshape(-1, 1))
            LdotN = LdotN*D
        LdotN = LdotN.clip(min=0)

        spec_color = row_mask_sum(incoming_light * ref_weight * LdotN, ray_mask) / row_mask_sum(LdotN * ref_weight, ray_mask).clip(min=1e-8).mean(dim=-1, keepdim=True)
        # ic((row_mask_sum(ref_weight * LdotN, ray_mask) / row_mask_sum(LdotN, ray_mask).clip(min=1e-8)).max())
        # spec_color = row_mask_sum(incoming_light * ref_weight * LdotN, ray_mask) / row_mask_sum(LdotN, ray_mask).clip(min=1e-8).mean(dim=-1, keepdim=True)

        return spec_color
