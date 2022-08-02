import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .render_modules import positional_encoding, str2fn
from icecream import ic

def schlick(f0, n, l):
    return f0 + (1-f0)*(1-(n*l).sum(dim=-1, keepdim=True).clip(min=1e-20))**5

def normalize(x):
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True)+1e-8)

def ggx_dist(NdotH, roughness):
    a2 = roughness**2
    return a2 / np.pi / ((NdotH**2*(a2-1)+1)**2).clip(min=1e-8)

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

class GGXSampler:
    def __init__(self, num_samples) -> None:
        self.sampler = torch.quasirandom.SobolEngine(dimension=2)
        self.num_samples = num_samples
        self.angs = self.sampler.draw(num_samples*2)

    def draw(self, B, num_samples):
        angs = self.angs.reshape(1, 2*self.num_samples, 2)[:, :num_samples, :].expand(B, num_samples, 2)
        # add random offset
        offset = torch.rand(B, 1, 2)
        angs = (angs + offset) % 1
        return angs

    def sample(self, num_samples, refdirs, viewdir, normal, roughness):
        # viewdir: (refdirs, 3)
        # viewdir: (B, 3)
        # normal: (B, 3)
        # roughness: B
        device = normal.device
        B = normal.shape[0]
        # adapated from learnopengl.com
        # a = (roughness*roughness).reshape(B, 1).expand(B, num_samples).reshape(-1)
        a = (roughness**2).reshape(B, 1)#.expand(B, num_samples).reshape(-1)
        angs = self.draw(B, num_samples).to(device)
	    
        phi = 2.0 * math.pi * angs[..., 0]
        cosTheta2 = ((1.0 - angs[..., 1]) / (1.0 + (a - 1.0) * angs[..., 1]).clip(min=1e-8))
        cosTheta = torch.sqrt(cosTheta2.clip(min=1e-8))
        sinTheta = torch.sqrt((1.0 - cosTheta2).clip(min=1e-8))

        H = torch.stack([
            torch.cos(phi)*sinTheta,
            torch.sin(phi)*sinTheta,
            cosTheta,
        ], dim=-1).reshape(B, num_samples, 3)
	    
        # from tangent-space vector to world-space sample vector
        # note: it's free to expand
        z_up = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, 3).expand(B, 3)
        x_up = torch.tensor([1.0, 0.0, 0.0], device=device).reshape(1, 3).expand(B, 3)
        up = torch.where(normal[:, 2:3] < 0.9999, z_up, x_up)
        tangent = torch.linalg.cross(up, normal)
        bitangent = torch.linalg.cross(normal, tangent)

        sampleVec = tangent.unsqueeze(1) * H[..., 0:1] + bitangent.unsqueeze(1) * H[..., 1:2] + normal.unsqueeze(1) * H[..., 2:3]
        sampleVec = normalize(sampleVec)
        L = normalize(2.0 * (viewdir.unsqueeze(1) * sampleVec).sum(dim=-1, keepdim=True) * sampleVec - viewdir.unsqueeze(1))
        L[:, 0] = refdirs.reshape(-1, 3)

        # calculate mipval, which will be used to calculate the mip level
        half = normalize(L + viewdir.reshape(-1, 1, 3))

        NdotH = (half * normal.reshape(-1, 1, 3)).sum(dim=-1).clip(min=1e-8)
        HdotV = (half * viewdir.reshape(-1, 1, 3)).sum(dim=-1).clip(min=1e-8)
        D = ggx_dist(NdotH, roughness.reshape(-1, 1))
        pdf = D * NdotH / 4 / HdotV
        mipval = 1 / (num_samples * pdf + 1e-6)
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

    def forward(self, incoming_light, V, L, N, features, matprop, mask, ray_mask):
        # V: (B, 3)-viewdirs, the outgoing light direction
        # L: (B, M, 3) incoming light direction. bounce_rays
        # N: (B, 3) outward normal
        # matprop: dictionary of attributes
        # mask: mask for matprop
        half = normalize(L + V.reshape(-1, 1, 3))

        cos_lamb = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        cos_view = (V.reshape(-1, 1, 3) * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
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
        alph = matprop['roughness'][mask].reshape(-1, 1, 1)
        a2 = alph**2
        # k = alph**2 / 2 # ibl
        # a2 = alph**2
        D_ggx = ggx_dist(cos_half, alph)
        k = (alph+1)**2 / 8 # direct lighting
        G_schlick_smith = 1 / ((cos_view*(1-k)+k)*(cos_lamb*(1-k)+k)).clip(min=1e-8)
        
        # f_s = D_ggx.clip(min=0, max=1) * G_schlick_smith.clip(min=0, max=1) / (4 * cos_lamb * cos_view).clip(min=1e-8)
        # f_s = D_ggx / (4 * cos_lamb * cos_view).clip(min=1e-8)

        f_s = D_ggx * G_schlick_smith / 4
        # f_s = D_ggx / (4 * cos_lamb * cos_view).clip(min=1e-8)


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
        # ref_weight = (1-ref_rough) * cos_refl + ref_rough * cos_lamb
        ref_weight = brdf * cos_lamb
        spec_color = (incoming_light * ref_weight).sum(dim=1)
        return spec_color

class MLPBRDFCos(torch.nn.Module):
    def __init__(self, in_channels, feape=6, featureC=128, num_layers=2, activation='sigmoid', lr=1e-3):
        super().__init__()
        self.in_mlpC = 2*feape*in_channels + in_channels + 4
        self.lr = lr
        self.feape = feape
        self.activation = str2fn(activation)
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
                torch.nn.Linear(featureC, 6),
            )
            # self.mlp.apply(self.init_weights)
        else:
            self.mlp = torch.nn.Identity()

    def forward(self, incoming_light, V, L, N, features, matprop, mask, ray_mask):
        roughness = matprop['roughness'][mask]
        D = features.shape[-1]
        n, m, _ = L.shape
        features = features.reshape(n, 1, D).expand(n, m, D).reshape(-1, D)
        eroughness = roughness.expand(n, m).reshape(-1, 1)

        B, M, _ = L.shape
        half = normalize(L + V.reshape(-1, 1, 3))

        cos_lamb = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        cos_view = (V.reshape(-1, 1, 3) * N).sum(dim=-1, keepdim=True).clip(min=1e-8).expand(n, m, 1)
        cos_half = (half * N).sum(dim=-1, keepdim=True)

        # ic(features.shape, eroughness.shape, cos_lamb.shape, cos_view.shape, cos_half.shape)
        indata = [features, eroughness, cos_lamb.reshape(-1, 1), cos_view.reshape(-1, 1), cos_half.reshape(-1, 1)]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]

        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in).reshape(n, m, -1)
        mlp_out = self.activation(mlp_out)
        ref_weight = mlp_out[:, :, :3]
        offset = mlp_out[:, :, 3:6]

        ref_weight = ref_weight / m
        # ref_weight = torch.softmax(ref_weight, dim=1)
        offset = offset / m
        spec_color = (ray_mask * incoming_light * ref_weight + offset).sum(dim=1) / ray_mask.sum(dim=1)
        return spec_color


class MLPBRDF(torch.nn.Module):
    def __init__(self, in_channels, v_encoder=None, n_encoder=None, l_encoder=None, feape=6, featureC=128, num_layers=2,
                 mul_ggx=False, activation='sigmoid', use_roughness=False, lr=1e-4, detach_roughness=False):
        super().__init__()

        self.in_channels = in_channels
        self.use_roughness = use_roughness
        self.detach_roughness = detach_roughness
        self.mul_ggx = mul_ggx
        self.in_mlpC = 2*feape*in_channels + in_channels + (1 if use_roughness else 0) + 3
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
                    ] for _ in range(num_layers)], []),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(featureC, 6),
            )
            self.mlp.apply(self.init_weights)
        else:
            self.mlp = torch.nn.Identity()
        self.activation = str2fn(activation)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def forward(self, incoming_light, V, L, N, features, matprop, mask, ray_mask):
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
        eroughness = roughness.expand(n, m).reshape(-1, 1)
        indata = [features]
        half = normalize(L + V.reshape(-1, 1, 3))

        cos_lamb = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        cos_view = (V.reshape(-1, 1, 3) * N).sum(dim=-1, keepdim=True).clip(min=1e-8).expand(n, m, 1)
        cos_half = (half * N).sum(dim=-1, keepdim=True)

        # ic(features.shape, eroughness.shape, cos_lamb.shape, cos_view.shape, cos_half.shape)
        # a = 0.2
        # v = a/(cos_half.reshape(-1, 1).abs()+a)
        indata = [features, cos_lamb.reshape(-1, 1), cos_view.reshape(-1, 1), cos_half.reshape(-1, 1).abs()]
        # indata = [features]
        if self.detach_roughness:
            eroughness = eroughness.detach()
        if self.use_roughness:
            indata.append(eroughness)

        V = V.reshape(n, 1, 3).expand(L.shape).reshape(-1, 3)
        N = N.expand(n, m, 3).reshape(-1, 3)
        L = L.reshape(-1, 3)
        B = V.shape[0]

        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.v_encoder is not None:
            indata += [self.v_encoder(V, eroughness).reshape(B, -1), V]
        if self.n_encoder is not None:
            indata += [self.n_encoder(N, eroughness).reshape(B, -1), N]
        if self.l_encoder is not None:
            indata += [self.l_encoder(L, eroughness).reshape(B, -1), L]

        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in).reshape(n, m, -1)
        mlp_out = self.activation(mlp_out)
        ref_weight = mlp_out[:, :, :3]
        offset = mlp_out[:, :, 3:6]

        if self.mul_ggx:
            alph = matprop['roughness'][mask].reshape(-1, 1, 1)
            half = normalize(L + V)
            cos_half = (half * N).sum(dim=-1, keepdim=True).reshape(n, m, 1)
            a2 = alph**2
            D_ggx = a2 / np.pi / ((cos_half**2*(a2-1)+1)**2).clip(min=1e-8)

            ref_weight = ref_weight*D_ggx

        # ref_weight = ref_weight / (ref_weight.sum(dim=1, keepdim=True).mean(dim=2, keepdim=True)+1e-8)
        # ref_weight = ref_weight * ray_mask / ray_mask.sum(dim=1, keepdim=True)
        ref_weight = ref_weight / m
        offset = offset / m
        spec_color = (ray_mask * incoming_light * ref_weight + offset).sum(dim=1) / ray_mask.sum(dim=1)
        return spec_color

class LightTinter(torch.nn.Module):
    def __init__(self, in_channels, v_encoder=None, n_encoder=None, l_encoder=None, feape=6, featureC=128, num_layers=2,
                 mul_ggx=False, activation='sigmoid', use_roughness=False, lr=1e-4, pbr=False):
        super().__init__()

        self.in_channels = in_channels
        self.use_roughness = use_roughness
        self.mul_ggx = mul_ggx
        self.in_mlpC = 2*feape*in_channels + in_channels + (1 if use_roughness else 0) + 2
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
                    ] for _ in range(num_layers)], []),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(featureC, 6),
            )
            self.mlp.apply(self.init_weights)
        else:
            self.mlp = torch.nn.Identity()
        self.activation = str2fn(activation)
        self.pbr = pbr

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def forward(self, incoming_light, V, L, N, features, matprop, mask, ray_mask):
        # V: (n, 3)-viewdirs, the outgoing light direction
        # L: (n, m, 3) incoming light direction. bounce_rays
        # N: (n, 1, 3) outward normal
        # features: (B, D)
        # matprop: dictionary of attributes
        # mask: mask for matprop
        roughness = matprop['roughness'][mask]
        D = features.shape[-1]
        indata = [features]
        cos_lamb = (L[:, 0] * N.reshape(-1, 3)).sum(dim=-1, keepdim=True).clip(min=1e-8)
        cos_view = (V.reshape(-1, 3) * N.reshape(-1, 3)).sum(dim=-1, keepdim=True).clip(min=1e-8)

        # ic(features.shape, eroughness.shape, cos_lamb.shape, cos_view.shape, cos_half.shape)
        # a = 0.2
        # v = a/(cos_half.reshape(-1, 1).abs()+a)
        indata = [features, cos_lamb.reshape(-1, 1), cos_view.reshape(-1, 1)]
        # indata = [features]
        if self.use_roughness:
            indata.append(roughness)

        B = V.shape[0]

        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.v_encoder is not None:
            indata += [self.v_encoder(V, roughness).reshape(B, -1), V]
        if self.n_encoder is not None:
            indata += [self.n_encoder(N, roughness).reshape(B, -1), N]
        if self.l_encoder is not None:
            indata += [self.l_encoder(L[:, 0], roughness).reshape(B, -1), L[:, 0]]

        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        tint = self.activation(mlp_out[..., :3])
        add = self.activation(mlp_out[..., 3:])

        if self.pbr:
            half = normalize(L + V.reshape(-1, 1, 3))

            cos_lamb = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
            cos_view = (V.reshape(-1, 1, 3) * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
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

            alph = matprop['roughness'][mask].reshape(-1, 1, 1)
            k = alph**2 / 2 # ibl
            # a2 = alph**2
            D_ggx = ggx_dist(cos_half, alph)
            # k = (alph+1)**2 / 8 # direct lighting
            G_schlick_smith = 1 / ((cos_view*(1-k)+k)*(cos_lamb*(1-k)+k)).clip(min=1e-8)
            
            # f_s = D_ggx.clip(min=0, max=1) * G_schlick_smith.clip(min=0, max=1) / (4 * cos_lamb * cos_view).clip(min=1e-8)
            # f_s = D_ggx / (4 * cos_lamb * cos_view).clip(min=1e-8)

            f_s = D_ggx * G_schlick_smith / 4
            # f_s = D_ggx / (4 * cos_lamb * cos_view).clip(min=1e-8)


            # the diffuse light is covered by other components of the rendering equation
            # ic(k_d.mean(dim=1).mean(dim=0), k_s.mean(dim=1).mean(dim=0))
            # brdf = k_d*f_d + k_s*f_s
            albedo = matprop['albedo'][mask].reshape(-1, 1, 3)
            brdf = k_d*albedo + k_s*f_s
            brdf *= ray_mask
            # normalize probabilities so they sum to 1. the rgb dims represent the spectra in equal parts.
            # brdf = brdf / brdf.sum(dim=1, keepdim=True).mean(dim=2, keepdim=True)
            # brdf = k_s * f_s

            # cos_refl = (noise_rays * refdirs[full_bounce_mask].reshape(-1, 1, 3)).sum(dim=-1, keepdim=True).clip(min=0)
            # cos_refl = (bounce_rays[..., 3:6] * refdirs[full_bounce_mask].reshape(-1, 1, 3)).sum(dim=-1, keepdim=True).clip(min=0)
            # ref_rough = roughness[bounce_mask].reshape(-1, 1)
            # phong shading?
            # ref_weight = (1-ref_rough) * cos_refl + ref_rough * cos_lamb
            ref_weight = brdf * cos_lamb
            spec_light = (incoming_light * ref_weight * ray_mask).sum(dim=1) / ray_mask.sum(dim=1)
        else:
            spec_light = (ray_mask*incoming_light).sum(dim=1) / ray_mask.sum(dim=1)
        spec_color = tint * spec_light + add
        return spec_color

