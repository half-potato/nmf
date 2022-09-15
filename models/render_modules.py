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
import cv2
from .grid_sample_Cinf import gkern

def str2fn(name):
    if name == 'sigmoid':
        return torch.nn.Sigmoid()
    elif name == 'softplus':
        return torch.nn.Softplus()
    elif name == 'identity':
        return torch.nn.Identity()
    elif name == 'clamp':
        return Clamp(0, 1)
    else:
        raise Exception(f"Unknown function {name}")

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

class Clamp(torch.nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return x.clamp(self.min, self.max)

class IPE(torch.nn.Module):
    def __init__(self, max_degree=8, in_dim=3) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.in_dim = in_dim

    def dim(self):
        return 2 * self.in_dim * self.max_degree

    def forward(self, viewdirs, roughness, **kwargs):
        size = roughness.reshape(-1, 1).expand(viewdirs.shape)
        return safemath.integrated_pos_enc((viewdirs, size), 0, self.max_degree)

class PE(torch.nn.Module):
    def __init__(self, max_degree=8, in_dim=3) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.in_dim = in_dim

    def dim(self):
        return 2 * self.in_dim * self.max_degree

    def forward(self, x, roughness, **kwargs):
        return positional_encoding(x, self.max_degree)

class VisibilityMLP(torch.nn.Module):
    def __init__(self, in_channels, view_encoder=None, feape=2, featureC=128, num_layers=4, lr=1e-3):
        super().__init__()

        self.lr = lr
        self.in_mlpC = 3
        if feape > -1:
            self.in_mlpC += 2*feape*in_channels + in_channels
        self.view_encoder = view_encoder
        if view_encoder is not None:
            self.in_mlpC += self.view_encoder.dim()
        self.feape = feape

        self.mlp = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(self.in_mlpC),
            torch.nn.Linear(self.in_mlpC, featureC),
            # torch.nn.BatchNorm1d(featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                    # torch.nn.BatchNorm1d(featureC),
                ] for _ in range(num_layers-2)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 2)
        )
        torch.nn.init.constant_(self.mlp[-1].bias, -2)
        self.mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def mask(self, norm_ray_origins, viewdirs, world_bounces, features):
        eterm, sigvis = self.visibility_module(norm_ray_origins, viewdirs, features)
        p = max(min(1 - world_bounces / sigvis.numel(), 1.0), 0.0)
        t = torch.quantile(sigvis.flatten(), p).clip(min=0.9)
        vis_mask = sigvis > t
        return vis_mask

    def update(self, norm_ray_origins, viewdirs, app_features, termination, bgvisibility):
        # bgvisibility is 1 if it reaches the BG and 0 if not
        eterm, sigvis = self.visibility_module(norm_ray_origins, viewdirs, app_features)
        # loss = ((termination - eterm)**2 + (sigvis-visibility.float())**2).sum()
        loss = ((sigvis-(~visibility).float())**2).mean()
        return loss

    def forward(self, pts, viewdirs, features, **kwargs):
        B = pts.shape[0]
        pts = pts[..., :3]


        indata = [viewdirs]
        if self.feape > -1:
            indata.append(features)
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            ise_enc = self.view_encoder(viewdirs, torch.tensor(20, device=pts.device).expand(B)).reshape(B, -1)
            indata += [ise_enc]

        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        sigvis = torch.sigmoid(out[..., 0])
        eterm = torch.exp(out[..., 1])

        return eterm, sigvis

class MLPRender_FP(torch.nn.Module):
    in_channels: int
    feape: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, view_encoder=None, ref_encoder=None, feape=6, featureC=128, num_layers=4, activation='softplus', lr=1e-3):
        super().__init__()

        self.lr = lr
        self.ref_encoder = ref_encoder
        self.in_mlpC = 3 + 1
        if feape > -1:
            self.in_mlpC += 2*feape*in_channels + in_channels
        self.view_encoder = view_encoder
        if view_encoder is not None:
            self.in_mlpC += self.view_encoder.dim()
        if ref_encoder is not None:
            self.in_mlpC += self.ref_encoder.dim()
        self.feape = feape

        self.mlp = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(self.in_mlpC),
            torch.nn.Linear(self.in_mlpC, featureC),
            # torch.nn.BatchNorm1d(featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                    # torch.nn.BatchNorm1d(featureC),
                ] for _ in range(num_layers-2)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3)
        )
        self.activation = str2fn(activation)
        torch.nn.init.constant_(self.mlp[-1].bias, -2)
        self.mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def forward(self, pts, viewdirs, features, refdirs, roughness, viewdotnorm, **kwargs):
        B = pts.shape[0]
        pts = pts[..., :3]


        indata = [refdirs, viewdotnorm]
        if self.feape > -1:
            indata.append(features)
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            ise_enc = self.view_encoder(viewdirs, torch.tensor(20, device=pts.device)).reshape(B, -1)
            indata += [ise_enc]
        if self.ref_encoder is not None:
            ise_enc = self.ref_encoder(refdirs, roughness).reshape(B, -1)
            indata += [ise_enc]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = self.activation(rgb)

        return rgb

class PassthroughDiffuse(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.allocation = 8
        self.lr = 0

    def forward(self, pts, viewdirs, features, **kwargs):
        B = pts.shape[0]
        mlp_out = features
        # max 0.5 roughness
        i = 0
        diffuse = torch.sigmoid(mlp_out[..., :i+3]-3)
        i += 3
        roughness = torch.sigmoid(mlp_out[..., i:i+1]+2).clip(min=1e-2)/2
        i += 1
        ambient = torch.sigmoid(mlp_out[..., i:i+1]-2)
        i += 1
        tint = torch.sigmoid(mlp_out[..., i:i+3])
        i += 3
        return diffuse, tint, dict(
            ambient = ambient,
            diffuse = diffuse,
            roughness = roughness,
        )

class MLPDiffuse(torch.nn.Module):
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, pospe=12, view_encoder=None, feape=6, featureC=128, num_layers=0, unlit_tint=False, lr=1e-4):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 3 + 2*max(feape, 0)*in_channels + in_channels if feape >= 0 else 0
        self.unlit_tint = unlit_tint
        self.lr = lr
        self.allocation = 0

        self.view_encoder = view_encoder
        if view_encoder is not None:
            self.in_mlpC += self.view_encoder.dim() + 3
        self.feape = feape
        self.pospe = pospe
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
                torch.nn.Linear(featureC, 20),
            )
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
            self.mlp.apply(self.init_weights)
        else:
            self.mlp = torch.nn.Identity()
        # to the neural network, roughness is unitless
        self.max_roughness = 1
        self.max_refraction_index = 2
        self.min_refraction_index = 1

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, pts, viewdirs, features, **kwargs):
        B = pts.shape[0]
        size = pts[..., 3:4].expand(pts[..., :3].shape)
        pts = pts[..., :3]
        indata = [pts]
        if self.pospe > 0:
            indata += [safemath.integrated_pos_enc((pts, size), 0, self.pospe)]

        if self.feape >= 0:
            indata.append(features)
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            indata += [self.view_encoder(viewdirs, torch.tensor(20.0, device=pts.device)).reshape(B, -1), viewdirs]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        rgb = torch.sigmoid(mlp_out)

        # ambient = F.softplus(mlp_out[..., 6:7]-3)
        ambient = torch.sigmoid(mlp_out[..., 6:7]-2)
        refraction_index = F.softplus(mlp_out[..., 7:8]-1) + self.min_refraction_index
        reflectivity = 50*F.softplus(mlp_out[..., 8:9])
        # roughness = F.softplus(mlp_out[..., 10:11]-1)
        # max 0.5 roughness
        roughness = torch.sigmoid(mlp_out[..., 10:11]).clip(min=1e-2)/2
        f0 = torch.sigmoid(mlp_out[..., 11:14])
        # albedo = F.softplus(mlp_out[..., 14:17]-2)
        albedo = torch.sigmoid(mlp_out[..., 14:17])
        ratio_diffuse = rgb[..., 9:10]
        if self.unlit_tint:
            h = mlp_out[..., 3]
            t = mlp_out[..., 4]
            sphere = torch.stack([
                torch.cos(h)*torch.cos(t),
                torch.sin(h)*torch.cos(t),
                torch.sin(t),
            ], dim=-1)
            tint = sphere/2 - 0.5
        else:
            # tint = F.softplus(mlp_out[..., 3:6])
            tint = torch.sigmoid(mlp_out[..., 3:6])
        # diffuse = rgb[..., :3]
        # tint = F.softplus(mlp_out[..., 3:6])
        diffuse = torch.sigmoid(mlp_out[..., :3]-1)

        # ic(f0)
        return diffuse, tint, dict(
            refraction_index = refraction_index,
            ratio_diffuse = ratio_diffuse,
            reflectivity = reflectivity,
            ambient = ambient,
            albedo=albedo,
            diffuse = diffuse,
            roughness = roughness,
            f0 = f0,
            tint=tint,
        )

class DeepMLPNormal(torch.nn.Module):
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int
    def __init__(self, pospe=16, in_channels=0, featureC=128, num_layers=2, lr=1e-4):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 3
        self.pospe = pospe
        self.lr = lr
        self.allocation = 0

        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_layers-2)], []),
            torch.nn.Tanh(),
            torch.nn.Linear(featureC, 3),
        )
        self.mlp0.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
            # m.bias.data.fill_(0.01)

    def forward(self, pts, features, **kwargs):
        size = pts[..., 3:4].expand(pts[..., :3].shape)
        pts = pts[..., :3]
        indata = [pts]
        if self.pospe > 0:
            # indata += [positional_encoding(pts, self.pospe)]
            indata += [safemath.integrated_pos_enc((pts, size), 0, self.pospe)]
        x0 = torch.cat(indata, dim=-1)
        x1 = self.mlp0(x0)

        # x2 = torch.cat([x0, x1], dim=-1)
        # x3 = self.mlp1(x2)
        # normals = torch.sin(x1)
        normals = x1
        normals = normals / torch.norm(normals, dim=-1, keepdim=True)

        return normals

class MLPNormal(torch.nn.Module):
    in_channels: int
    feape: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, pospe=6, feape=6, featureC=128, num_layers=2, lr=1e-4):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 2*max(feape, 0)*in_channels + 3 + in_channels if feape >= 0 else 0
        self.pospe = pospe
        self.feape = feape
        self.lr = lr
        self.allocation = 0

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_layers-2)], []),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Tanh(),
            torch.nn.Linear(featureC, 3, bias=False)
        )

        # self.mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, pts, features, **kwargs):
        size = pts[..., 3:4].expand(pts[..., :3].shape)
        pts = pts[..., :3]
        indata = [pts]
        if self.feape >= 0:
            indata.append(features)

        if self.pospe > 0:
            indata += [safemath.integrated_pos_enc((pts, size), 0, self.pospe)]
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

class AppDimNormal(torch.nn.Module):                                                                                                                                                                               
    def __init__(self, in_channels=0, activation=torch.nn.Identity):                                                                                                                                                                             
        super().__init__()                                                                                                                                                                                         
        self.activation = activation()
        self.lr = 1
        self.allocation = 3
                                                                                                                                                                                                                   
    def forward(self, pts, features, **kwargs):                                                                                                                                                                    
        start_ind = 0
        # raw_norms = features[..., start_ind:start_ind+3]                                                                                                                                                           
        raw_norms = features[..., start_ind:start_ind+3]
        # raw_norms = 2*torch.sigmoid(raw_norms)-1
        raw_norms = self.activation(raw_norms)
        normals = raw_norms / (torch.norm(raw_norms, dim=-1, keepdim=True) + 1e-8)
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
