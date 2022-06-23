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
from typing import List

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


class BackgroundRender(torch.nn.Module):
    bg_rank: int
    bg_resolution: List[int]
    def __init__(self, bg_rank, bg_resolution=512, view_encoder=None, featureC=128, num_layers=2):
        super().__init__()
        self.bg_mat = nn.Parameter(0.1 * torch.randn((1, bg_rank, bg_resolution*2, bg_resolution))) # [1, R, H, W]
        self.view_encoder = view_encoder
        self.bg_rank = bg_rank
        d = self.view_encoder.dim() if self.view_encoder is not None else 0
        if num_layers == 0 and bg_rank == 3:
            self.bg_net = nn.Sigmoid()
        else:
            self.bg_net = nn.Sequential(
                nn.Linear(bg_rank+d, featureC, bias=False),
                *sum([[
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(featureC, featureC, bias=False)
                    ] for _ in range(num_layers-2)], []),
                torch.nn.ReLU(inplace=True),
                nn.Linear(featureC, 3, bias=False),
                nn.Sigmoid()
            )

    @torch.no_grad()
    def upsample(self, bg_resolution):
        self.bg_mat = torch.nn.Parameter(
            F.interpolate(self.bg_mat.data, size=(bg_resolution*2, bg_resolution), mode='bilinear', align_corners=True)
        )
        
    def forward(self, viewdirs):
        B = viewdirs.shape[0]
        a, b, c = viewdirs[:, 0:1], viewdirs[:, 1:2], viewdirs[:, 2:3]
        norm2d = torch.sqrt(a**2+b**2)
        phi = safemath.atan2(b, a)
        theta = safemath.atan2(c, norm2d)
        x = torch.cat([
            (phi % (2*np.pi) - np.pi) / np.pi,
            theta/np.pi/2,
        ], dim=1).reshape(1, 1, -1, 2)
        emb = F.grid_sample(self.bg_mat, x, mode='bicubic', align_corners=True)
        emb = emb.reshape(self.bg_rank, -1).T
        # return torch.sigmoid(emb)
        if self.view_encoder is not None:
            shemb = self.view_encoder(viewdirs, torch.tensor(20.0, device=viewdirs.device)).reshape(B, -1)
            emb = torch.cat([emb, shemb], dim=1)
        return self.bg_net(emb)


class MLPRender_FP(torch.nn.Module):
    in_channels: int
    feape: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, view_encoder=None, ref_encoder=None, feape=6, featureC=128, num_layers=4):
        super().__init__()

        self.ref_encoder = ref_encoder
        self.in_mlpC = 2*feape*in_channels + 6 + in_channels
        self.view_encoder = view_encoder
        if view_encoder is not None:
            self.in_mlpC += self.view_encoder.dim()
        if ref_encoder is not None:
            self.in_mlpC += self.ref_encoder.dim()
        self.feape = feape

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_layers)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3)
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)
        self.mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def forward(self, pts, viewdirs, features, refdirs, roughness, **kwargs):
        B = pts.shape[0]
        pts = pts[..., :3]


        indata = [pts, features, refdirs]
        # if self.pospe > 0:
        #     indata += [positional_encoding(pts, self.pospe)]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            # indata += [positional_encoding(viewdirs, self.viewpe), viewdirs]
            ise_enc = self.view_encoder(viewdirs, torch.tensor(20, device=pts.device)).reshape(B, -1)
            indata += [ise_enc]
            # ise_enc = self.spherical_encoder(viewdirs, roughness).reshape(-1, (self.refpe+1)*4)
            # indata += [torch.sigmoid(ise_enc)]
        if self.ref_encoder is not None:
            # indata += [self.spherical_encoder(refdirs, 1/torch.sqrt(roughness+1e-6)).reshape(-1, (self.refpe+1)*4)/100]
            # roughness = torch.tensor(20, device=pts.device)
            ise_enc = self.ref_encoder(refdirs, roughness).reshape(B, -1)
            # ise_enc = self.spherical_encoder(refdirs, roughness).reshape(B, -1)
            indata += [ise_enc]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPDiffuse(torch.nn.Module):
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, pospe=12, view_encoder=None, feape=6, featureC=128, num_layers=0):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 3 + 2*feape*in_channels + in_channels

        self.view_encoder = view_encoder
        if view_encoder is not None:
            self.in_mlpC += self.view_encoder.dim()
        self.feape = feape
        self.pospe = pospe
        if num_layers > 0:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.in_mlpC, featureC),
                # torch.nn.ReLU(inplace=True),
                # torch.nn.Linear(featureC, featureC),
                *sum([[
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(featureC, featureC),
                    ] for _ in range(num_layers)], []),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(featureC, 8),
            )
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
        else:
            self.mlp = torch.nn.Identity()
        ang_roughness = 20/180*np.pi
        # self.max_roughness = 30/180*np.pi
        self.max_roughness = 40
        self.max_refraction_index = 2
        self.min_refraction_index = 1
        # x = ang_roughness / self.max_roughness
        # sigmoid_out = np.log(x/(1-x))
        # torch.nn.init.constant_(self.mlp[-1].bias[-1], sigmoid_out)

    def forward(self, pts, viewdirs, features, **kwargs):
        B = pts.shape[0]
        pts = pts[..., :3]
        indata = [features, pts]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            indata += [self.view_encoder(viewdirs, torch.tensor(20.0, device=pts.device)).reshape(B, -1)]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        rgb = torch.sigmoid(mlp_out)

        roughness = rgb[..., 6:7]*self.max_roughness
        refraction_index = F.softplus(mlp_out[..., 7:8]) + self.min_refraction_index
        tint = rgb[..., 3:6] 
        diffuse = rgb[..., :3] 

        return diffuse, tint, roughness, refraction_index

class MLPRender_Fea(torch.nn.Module):
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int
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
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int
    def __init__(self, pospe=16, featureC=128, num_layers=2):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 3
        self.pospe = pospe

        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_layers)], []),
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
    in_channels: int
    feape: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, pospe=6, feape=6, featureC=128, num_layers=2):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 2*feape*in_channels + 3 + in_channels
        self.pospe = pospe
        self.feape = feape

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_layers)], []),
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
