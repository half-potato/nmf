import torch
import torch.nn as nn
import torch.nn.functional as F
from .sh import eval_sh_bases
from math import pi
from icecream import ic
from . import safemath
import numpy as np
from models.ise import ISE, RandISE
from models.ish import ISH, RandISH

def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def spherical_encoding(refdirs, roughness, pe, ind_order=[0, 1, 2]):
    i, j, k = ind_order
    norm2d = torch.sqrt(refdirs[..., i]**2+refdirs[..., j]**2)
    refangs = torch.stack([
        safemath.atan2(refdirs[..., j], refdirs[..., i]) * norm2d,
        safemath.atan2(refdirs[..., k], norm2d),
    ], dim=-1)
    return [safemath.integrated_pos_enc((refangs[..., 0:1], roughness), 0, pe),
            safemath.integrated_pos_enc((refangs[..., 1:2], roughness), 0, pe)]

def normal_dist(x, sigma: float):
    SQ2PI = 2.50662827463
    return torch.exp(-(x/sigma)**2/2) / SQ2PI / sigma

def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb

class MLPRender_FP(torch.nn.Module):
    def __init__(self, in_channels, pospe=16, viewpe=6, feape=6, refpe=6, featureC=128):
        super().__init__()

        # self.spherical_encoder = ISH(refpe)
        self.vspherical_encoder = ISH(viewpe)
        # self.spherical_encoder = RandISE(512, 32)
        self.spherical_encoder = RandISE(128, 5)
        self.in_mlpC = 2*pospe*3 + 2*feape*in_channels + 6 + in_channels + self.spherical_encoder.dim() + self.vspherical_encoder.dim()
        # self.in_mlpC += 2 if refpe > 0 else 0
        # self.in_mlpC += 3 if viewpe > 0 else 0
        self.viewpe = viewpe
        self.refpe = refpe
        self.feape = feape
        self.pospe = pospe

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3)
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)
        self.mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear) and m.weight.shape[1] > 200:
            torch.nn.init.xavier_uniform_(m.weight, gain=0.2688)

    def forward(self, pts, viewdirs, features, refdirs, roughness, **kwargs):
        B = pts.shape[0]
        pts = pts[..., :3]


        indata = [pts, features, refdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            # indata += [positional_encoding(viewdirs, self.viewpe), viewdirs]
            ise_enc = self.vspherical_encoder(viewdirs, torch.tensor(20, device=pts.device)).reshape(B, -1)
            indata += [ise_enc]
            # ise_enc = self.spherical_encoder(viewdirs, roughness).reshape(-1, (self.refpe+1)*4)
            # indata += [torch.sigmoid(ise_enc)]
        if self.refpe > 0:
            # indata += [self.spherical_encoder(refdirs, 1/torch.sqrt(roughness+1e-6)).reshape(-1, (self.refpe+1)*4)/100]
            # roughness = torch.tensor(20, device=pts.device)
            ise_enc = self.spherical_encoder(refdirs, roughness).reshape(B, -1)
            # ise_enc = self.spherical_encoder(refdirs, roughness).reshape(B, -1)
            indata += [ise_enc]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPDiffuse(torch.nn.Module):
    def __init__(self, in_channels, pospe=12, feape=6, featureC=128):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 3 + 2*feape*in_channels + in_channels
        self.feape = feape
        self.pospe = pospe
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 7),
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)
        ang_roughness = 20/180*np.pi
        # self.max_roughness = 30/180*np.pi
        self.max_roughness = 40
        # x = ang_roughness / self.max_roughness
        # sigmoid_out = np.log(x/(1-x))
        # torch.nn.init.constant_(self.mlp[-1].bias[-1], sigmoid_out)

    def forward(self, pts, features, **kwargs):
        pts = pts[..., :3]
        indata = [features, pts]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)
        roughness = rgb[..., 6:7]*self.max_roughness
        tint = rgb[..., 3:6] 
        diffuse = rgb[..., :3] 

        return diffuse, tint, roughness
class MLPRender_Fea(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, feape=6, refpe=6, featureC=128):
        super().__init__()

        self.in_mlpC = 2*refpe*3 + 2*viewpe*3 + 2*feape*in_channels + 3 + in_channels
        self.in_mlpC += 3 if refpe > 0 else 0
        self.viewpe = viewpe
        self.refpe = refpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(
            inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, refdirs=None, **kwargs):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        if self.refpe > 0:
            indata += [positional_encoding(refdirs, self.refpe), refdirs]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class DeepMLPNormal(torch.nn.Module):
    def __init__(self, pospe=16, featureC=128):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 3
        self.pospe = pospe

        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Linear(featureC, featureC),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3),
            # torch.nn.ReLU(inplace=True),
        )
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC+featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3)
        )
        self.mlp0.apply(self.init_weights)
        self.mlp1.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.01)

    def forward(self, pts, features, **kwargs):
        pts = pts[..., :3]
        indata = [pts]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        x0 = torch.cat(indata, dim=-1)
        x1 = self.mlp0(x0)
        # x2 = torch.cat([x0, x1], dim=-1)
        # x3 = self.mlp1(x2)
        normals = torch.sin(x1)
        normals = normals / torch.norm(normals, dim=-1, keepdim=True)

        return normals

class MLPNormal(torch.nn.Module):
    def __init__(self, in_channels, pospe=6, feape=6, featureC=128):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 2*feape*in_channels + 3 + in_channels
        self.pospe = pospe
        self.feape = feape

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3)
        )

    def forward(self, pts, features, **kwargs):
        pts = pts[..., :3]
        indata = [features, pts]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        # angles = self.mlp(mlp_in)
        # azi = angles[:, 0]
        # ele = angles[:, 1]
        # normals = torch.stack([
        #     torch.cos(ele) * torch.cos(azi),
        #     torch.cos(ele) * torch.sin(azi),
        #     -torch.sin(ele),
        # ], dim=1)

        normals = self.mlp(mlp_in)
        normals = normals / torch.norm(normals, dim=-1, keepdim=True)

        return normals

class MLPRender_PE(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, pospe=6, featureC=128):
        super().__init__()

        self.in_mlpC = (3+2*viewpe*3) + (3+2*pospe*3) + in_channels
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(
            inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, featureC=128):
        super().__init__()

        self.in_mlpC = (3+2*viewpe*3) + in_channels
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(
            inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class BundleMLPRender(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, featureC=128, bundle_size=3):
        super().__init__()

        self.in_mlpC = (3+2*viewpe*3) + in_channels
        self.viewpe = viewpe
        self.bundle_size = bundle_size

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3*self.bundle_size**2)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(
            inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb.reshape(-1, self.bundle_size, self.bundle_size, 3)


class BundleSHRender_Fea(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, feape=6, featureC=128, bundle_size=3, ray_up_pe=2):
        super().__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*in_channels + \
            3 + in_channels + 1 + 2*ray_up_pe*3
        self.viewpe = viewpe
        self.feape = feape
        self.ray_up_pe = ray_up_pe
        self.bundle_size = bundle_size
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3*self.bundle_size**2)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(
            inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, bundle_size_w, ray_up):
        indata = [features, viewdirs, bundle_size_w]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.ray_up_pe > 0:
            indata += [positional_encoding(ray_up, self.ray_up_pe)]

        # (..., (deg+1) ** 2)
        sh_mult = eval_sh_bases(self.degree, viewdirs)[:, None]
        rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
        rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)

        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        rgb = torch.sigmoid(mlp_out)

        return rgb.reshape(-1, self.bundle_size, self.bundle_size, 3)


class BundleMLPRender_Fea(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, feape=6, featureC=128, bundle_size=3, ray_up_pe=2):
        super().__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*in_channels + \
            3 + in_channels + 1 + 2*ray_up_pe*3
        self.viewpe = viewpe
        self.feape = feape
        self.ray_up_pe = ray_up_pe
        self.bundle_size = bundle_size
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3*self.bundle_size**2)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(
            inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, bundle_size_w, ray_up, roughness=None):
        indata = [features, viewdirs, bundle_size_w]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        if self.ray_up_pe > 0:
            indata += [positional_encoding(ray_up, self.ray_up_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        rgb = torch.sigmoid(mlp_out)

        return rgb.reshape(-1, self.bundle_size, self.bundle_size, 3)


class LearnableSphericalEncoding(torch.nn.Module):
    def __init__(self, out_channels, out_res):
        super().__init__()
        # out_res is the number of points used to represent the sphere
        # out channels is the number of channels per a point
        self.out_res = out_res
        self.out_channels = out_channels

        # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069
        if out_res < 24:
            eps = 0.33
        elif out_res < 177:
            eps = 1.33
        elif out_res < 890:
            eps = 3.33

        weights = torch.rand((1, out_res, out_channels))
        # weights = torch.ones((1, out_res, out_channels))
        self.register_parameter('weights', torch.nn.Parameter(weights))

        indices = torch.arange(0, out_res, dtype=float)
        goldenRatio = (1 + 5**0.5) / 2

        phi = torch.arccos(1 - 2*(indices+eps)/(out_res-1+2*eps))
        theta = 2*pi * indices / goldenRatio

        x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi);
        self.register_buffer('sphere_pos', torch.stack([x, y, z], dim=0).float())

    def forward(self, vec, sigma):
        # vec: N, 3 normal vectors representing input directions
        # output: N, C

        # cos_dist: N, M
        cos_dist = (vec @ self.sphere_pos).clip(min=-1+1e-5, max=1-1e-5)
        # weights: 1, M, C
        # output: (N, 1, M) @ (1, M, C) -> (N, 1, C)
        prob = normal_dist(torch.arccos(cos_dist), sigma)
        prob /= (prob.sum(dim=1, keepdim=True) + 1e-8)
        output = torch.matmul(prob.unsqueeze(1), self.weights)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # # ax.scatter(self.sphere_pos[0].cpu(), self.sphere_pos[1].cpu(), self.sphere_pos[2].cpu(), c=prob[0].detach().cpu())
        # ic(self.weights.max(), sigma.min(), sigma.max())
        # col = self.weights[0].detach().cpu()
        # ax.scatter(self.sphere_pos[0].cpu(), self.sphere_pos[1].cpu(), self.sphere_pos[2].cpu(), c=torch.sigmoid(col))
        # plt.show()
        return output.squeeze(1)

class BundleDirectSphEncoding(torch.nn.Module):
    def __init__(self, in_channels, feape=6, featureC=128, bundle_size=3, ray_up_pe=2, sph_channels=3, sph_res=500):
        super().__init__()

        self.in_mlpC = 2*feape*in_channels + 3 + in_channels + 1 + 2*ray_up_pe*3
        self.feape = feape
        self.ray_up_pe = ray_up_pe
        self.bundle_size = bundle_size
        self.sph_enc = LearnableSphericalEncoding(sph_channels, sph_res)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3*self.bundle_size**2)
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, bundle_size_w, ray_up, roughness):
        # viewenc = self.sph_enc(viewdirs, 20/180*pi)
        viewenc = self.sph_enc(viewdirs, roughness.unsqueeze(1))
        indata = [features, viewdirs, bundle_size_w]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.ray_up_pe > 0:
            indata += [positional_encoding(ray_up, self.ray_up_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        # rgb = torch.sigmoid(mlp_out).reshape(-1, self.bundle_size, self.bundle_size, 3) + torch.sigmoid(viewenc).reshape(-1, 1, 1, 3)
        scaling = (roughness.unsqueeze(1)*180/pi / 10).clamp(min=1, max=20)
        rgb = torch.sigmoid(mlp_out).reshape(-1, self.bundle_size, self.bundle_size, 3) + (torch.sigmoid(viewenc)/scaling).reshape(-1, 1, 1, 3)

        return rgb

class BundleSphEncoding(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, feape=6, featureC=128, bundle_size=3, ray_up_pe=2, sph_channels=9, sph_res=64):
        super().__init__()

        self.in_mlpC = sph_channels + 2*viewpe*3 + 2*feape*in_channels + 3 + in_channels + 1 + 2*ray_up_pe*3
        self.viewpe = viewpe
        self.feape = feape
        self.ray_up_pe = ray_up_pe
        self.bundle_size = bundle_size
        self.sph_enc = LearnableSphericalEncoding(sph_channels, sph_res)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3*self.bundle_size**2)
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, bundle_size_w, ray_up, roughness):
        viewenc = self.sph_enc(viewdirs, roughness.reshape(-1, 1))
        indata = [features, viewenc, viewdirs, bundle_size_w]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        if self.ray_up_pe > 0:
            indata += [positional_encoding(ray_up, self.ray_up_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        rgb = torch.sigmoid(mlp_out)

        return rgb.reshape(-1, self.bundle_size, self.bundle_size, 3)

class BundleMLPRender_Fea_Grid(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, feape=6, featureC=128, bundle_size=3, extra=4, ray_up_pe=4, refpe=6):
        super().__init__()

        self.in_mlpC = 2*viewpe*3 + 3
        self.in_mlpC += 2*feape*in_channels + in_channels
        self.in_mlpC += 1 + 2*ray_up_pe*3 # ray up + size
        if refpe > 0:
            self.in_mlpC += 2*refpe*3 + 3 # ref dir
        self.viewpe = viewpe
        self.feape = feape
        self.refpe = refpe
        self.ray_up_pe = ray_up_pe
        self.bundle_size = bundle_size
        assert(extra == 4 or extra == 8 or extra == 9)
        self.extra = extra
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3*self.bundle_size**2+self.extra+1)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(
            inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)
        f_blur = torch.tensor([1, 2, 1]) / 4
        f_edge = torch.tensor([-1, 0, 1]) / 2
        dy = (f_blur[None, :] * f_edge[:, None]).reshape(1, 3, 3)
        dx = (f_blur[:, None] * f_edge[None, :]).reshape(1, 3, 3)

        self.register_buffer('dy', dy)
        self.register_buffer('dx', dx)

    def forward(self, pts, viewdirs, features, bundle_size_w, ray_up, refdirs=None):
        indata = [features, viewdirs, bundle_size_w]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        if self.refpe > 0:
            indata += [positional_encoding(refdirs, self.refpe), refdirs]
        if self.ray_up_pe > 0:
            indata += [positional_encoding(ray_up, self.ray_up_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        m = 3*self.bundle_size**2
        rgb = mlp_out[:, :m]
        extra = mlp_out[:, m:]
        roughness = torch.sigmoid(extra[:, 0])*60/180*pi
        rgb = torch.sigmoid(rgb)

        # rel_density = torch.tanh(extra)
        rel_density = extra[:, 1:]
        if self.extra == 4:
            left_density = torch.stack([
                rel_density[:, 0],
                torch.zeros_like(rel_density[:, 0]),
                rel_density[:, 1],
            ], dim=1)
            top_density = torch.stack([
                rel_density[:, 2],
                torch.zeros_like(rel_density[:, 0]),
                rel_density[:, 3],
            ], dim=1)
            # (-1, bundle_size, bundle_size)
            rel_density_grid = left_density.reshape(
                -1, 3, 1) + top_density.reshape(-1, 1, 3)
        elif self.extra == 8:
            rel_density_grid = torch.stack([
                rel_density[:, 0],
                rel_density[:, 1],
                rel_density[:, 2],
                rel_density[:, 3],
                torch.zeros_like(rel_density[:, 0]),
                rel_density[:, 4],
                rel_density[:, 5],
                rel_density[:, 6],
                rel_density[:, 7],
            ], dim=1).reshape(-1, 3, 3)
        else:
            rel_density_grid = rel_density.reshape(-1, 3, 3)


        dx = (self.dx * F.relu(rel_density_grid)).sum(-1).sum(-1)
        dy = (self.dy * F.relu(rel_density_grid)).sum(-1).sum(-1)
        normal = torch.stack([
            dx, dy, torch.ones_like(dx)
        ], dim=1)
        rgb = rgb.reshape(-1, self.bundle_size, self.bundle_size, 3)
        # rgb[:, 0, 0, :] = 0
        return rgb, rel_density_grid, roughness, normal

class BundleMLPRender_Fea_Normal(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, feape=6, featureC=128, bundle_size=3, ray_up_pe=4, refpe=6):
        super().__init__()

        self.in_mlpC = 2*viewpe*3 + 3
        self.in_mlpC += 2*feape*in_channels + in_channels
        self.in_mlpC += 1 + 2*ray_up_pe*3 # ray up + size
        if refpe > 0:
            self.in_mlpC += 2*refpe*3 + 3 # ref dir
        self.viewpe = viewpe
        self.feape = feape
        self.refpe = refpe
        self.ray_up_pe = ray_up_pe
        self.bundle_size = bundle_size
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3*self.bundle_size**2+3+1)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(
            inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, bundle_size_w, ray_up, refdirs=None):
        indata = [features, viewdirs, bundle_size_w]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        if self.refpe > 0:
            indata += [positional_encoding(refdirs, self.refpe), refdirs]
        if self.ray_up_pe > 0:
            indata += [positional_encoding(ray_up, self.ray_up_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        m = 3*self.bundle_size**2
        rgb = mlp_out[:, :m]
        extra = mlp_out[:, m:]
        roughness = torch.sigmoid(extra[:, 0])*60/180*pi
        rgb = torch.sigmoid(rgb)

        # rel_density = torch.tanh(extra)
        normal = extra[:, 1:]
        normal = normal / torch.norm(normal, dim=1, keepdim=True)
        rel_density_grid = torch.zeros((3, 3), device=pts.device)

        rgb = rgb.reshape(-1, self.bundle_size, self.bundle_size, 3)
        # rgb[:, 0, 0, :] = 0
        return rgb, rel_density_grid, roughness, normal
