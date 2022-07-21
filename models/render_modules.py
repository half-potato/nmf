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

class HierarchicalBG(torch.nn.Module):
    def __init__(self, bg_rank, unwrap_fn, bg_resolution=512, num_levels=2, featureC=128, num_layers=2):
        super().__init__()
        self.bg_resolution = bg_resolution
        self.num_levels = num_levels
        self.bg_mats = nn.ParameterList([
            # nn.Parameter(0.5 * torch.rand((1, bg_rank, 2**i * bg_resolution*unwrap_fn.H_mul, 2**i * bg_resolution*unwrap_fn.W_mul)))
            nn.Parameter(0.5 * torch.zeros((1, bg_rank, 4**i * bg_resolution*unwrap_fn.H_mul, 4**i * bg_resolution*unwrap_fn.W_mul)))
            for i in range(num_levels)])
        self.unwrap_fn = unwrap_fn
        self.bg_rank = bg_rank
        if num_layers == 0 and bg_rank == 3:
            # self.bg_net = nn.Softplus(beta=50)
            # self.bg_net = nn.ReLU()
            self.bg_net = nn.Identity()
        else:
            self.bg_net = nn.Sequential(
                nn.Linear(bg_rank, featureC, bias=False),
                *sum([[
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(featureC, featureC, bias=False)
                    ] for _ in range(num_layers-2)], []),
                torch.nn.ReLU(inplace=True),
                nn.Linear(featureC, 3, bias=False),
                nn.Softplus()
            )
        self.align_corners = False
        self.smoothing = 1

    @torch.no_grad()
    def save(self, path):
        bg_resolution = self.bg_mats[-1].shape[2] // self.unwrap_fn.H_mul
        bg_mats = []
        for i in range(self.num_levels):
            bg_mat = F.interpolate(self.bg_mats[i].data, size=(bg_resolution*self.unwrap_fn.H_mul, bg_resolution*self.unwrap_fn.W_mul), mode='bilinear', align_corners=self.align_corners)
            bg_mat = bg_mat / 2**i
            bg_mats.append(bg_mat)
        bg_mat = sum(bg_mats)
        im = (255*(self.bg_net(bg_mat)).clamp(0, 1)).short().permute(0, 2, 3, 1).squeeze(0)
        im = im.cpu().numpy()
        im = im[::-1].astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, im)

    @torch.no_grad()
    def upsample(self, bg_resolution):
        return
        self.bg_mats = nn.ParameterList([
                nn.Parameter(
                    F.interpolate(self.bg_mats[i].data, size=(2**i * bg_resolution*self.unwrap_fn.H_mul, 2**i * bg_resolution*self.unwrap_fn.W_mul), mode='bilinear', align_corners=self.align_corners)
                )
                for i in range(self.num_levels)
            ])
        
    def forward(self, viewdirs):
        B = viewdirs.shape[0]
        x = self.unwrap_fn(viewdirs)

        embs = []
        for i, bg_mat in enumerate(self.bg_mats):

            # smooth_kern = gkern(2*int(self.smoothing)+1, std=self.smoothing+1e-8, device=viewdirs.device)
            # s = smooth_kern.shape[-1]
            # smooth_mat = F.conv2d(bg_mat.permute(1, 0, 2, 3), smooth_kern.reshape(1, -1, s, s), stride=1, padding=s//2)
            #
            # emb = F.grid_sample(smooth_mat.permute(1, 0, 2, 3), x, mode='bilinear', align_corners=self.align_corners)

            emb = F.grid_sample(bg_mat, x, mode='bilinear', align_corners=self.align_corners)
            emb = emb.reshape(self.bg_rank, -1).T
            embs.append(emb / 2**i)
        emb = sum(embs)
        return self.bg_net(emb)


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
            # TODO REMOVE
            # self.bg_net = nn.Softplus()
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
        
    def forward(self, viewdirs):
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


class MLPRender_FP(torch.nn.Module):
    in_channels: int
    feape: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, view_encoder=None, ref_encoder=None, feape=6, featureC=128, num_layers=4, lr=1e-3):
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
                ] for _ in range(num_layers)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3)
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)
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
        # rgb = F.softplus(rgb)

        return rgb

class MLPDiffuse(torch.nn.Module):
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, pospe=12, view_encoder=None, feape=6, featureC=128, num_layers=0, unlit_tint=False):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 3 + 2*feape*in_channels + in_channels
        self.unlit_tint = unlit_tint

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
                    ] for _ in range(num_layers)], []),
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
        indata = [features, pts]
        if self.pospe > 0:
            indata += [safemath.integrated_pos_enc((pts, size), 0, self.pospe)]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            indata += [self.view_encoder(viewdirs, torch.tensor(20.0, device=pts.device)).reshape(B, -1), viewdirs]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        rgb = torch.sigmoid(mlp_out)

        ambient = F.softplus(mlp_out[..., 6:7])
        refraction_index = F.softplus(mlp_out[..., 7:8]-1) + self.min_refraction_index
        reflectivity = torch.sigmoid(mlp_out[..., 8:9])
        roughness = F.softplus(mlp_out[..., 10:11])
        f0 = torch.sigmoid(mlp_out[..., 11:12])
        # ambient = torch.sigmoid(mlp_out[..., 9:10])
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
            tint = rgb[..., 3:6]
        # diffuse = rgb[..., :3]
        # tint = F.softplus(mlp_out[..., 3:6])
        diffuse = torch.sigmoid(mlp_out[..., :3])

        # ic(f0)
        return diffuse, tint, dict(
            refraction_index = refraction_index,
            ratio_diffuse = ratio_diffuse,
            reflectivity = reflectivity,
            ambient = ambient,
            diffuse = diffuse,
            roughness = roughness,
            f0 = f0*0+1,
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

        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_layers)], []),
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

class BasisMLPNormal(torch.nn.Module):
    in_channels: int
    feape: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, num_basis=32, pospe=6, feape=6, featureC=128, num_layers=2, lr=1e-4):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 2*max(feape, 0)*in_channels + 3 + (in_channels if feape >= 0 else 0)
        self.pospe = pospe
        self.feape = feape
        self.lr = lr

        basis = torch.rand(num_basis, 3)*2-1
        basis = basis / torch.norm(basis, dim=-1, keepdim=True)
        # basis /= num_basis
        self.num_basis = num_basis
        # l = self.num_basis//4
        # basis[1*l:2*l] /= 32
        # basis[2*l:3*l] /= 512
        # basis[3*l:4*l] /= 2048
        self.register_parameter('basis', torch.nn.Parameter(basis))
        # self.register_buffer('basis', basis)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_layers)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, num_basis),
            torch.nn.Tanh(),
            # torch.nn.Softmax(dim=-1)
            # torch.nn.Sigmoid()
        )

        self.mlp.apply(self.init_weights)

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

        coeffs = self.mlp(mlp_in)
        normals = coeffs @ self.basis

        normals = normals / torch.norm(normals, dim=-1, keepdim=True)

        return normals

class MLPNormal(torch.nn.Module):
    in_channels: int
    feape: int
    featureC: int
    num_layers: int
    def __init__(self, in_channels, pospe=6, feape=6, featureC=128, num_layers=2, lr=1e-4):
        super().__init__()

        self.in_mlpC = 2*pospe*3 + 2*max(feape, 0)*in_channels + 3 + in_channels
        self.pospe = pospe
        self.feape = feape
        self.lr = lr

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(featureC, featureC),
                ] for _ in range(num_layers)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3)
        )

        self.mlp.apply(self.init_weights)

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
                                                                                                                                                                                                                   
    def forward(self, pts, features, **kwargs):                                                                                                                                                                    
        start_ind = 10                                                                                                                                                                                              
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
