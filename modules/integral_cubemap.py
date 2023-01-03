import warp as wp
import math
import torch
import torch.nn as nn
import nvdiffrast as nvdr

def tabularize_adjacencies():
    cache = None
    # TODO
    return cache

def project2face(xyz, face_ind):
    # TODO
    # xyz: (N, 3)
    x = xyz[:, 0].clone()
    y = xyz[:, 1].clone()
    z = xyz[:, 2].clone()

wp.init()
@wp.kernel
def index_cube_map(
    xyz: wp.array2d(dtype=float),
    ys: wp.array(dtype=float),
    ms: wp.array(dtype=float),
    idx: wp.array(dtype=float)):

    b = wp.tid()
    x = xyz[b, 0]
    y = xyz[b, 1]
    z = xyz[b, 2]
    ax = abs(x)
    ay = abs(y)
    az = abs(z)

    c = float(0.0)
    _idx = 0
    if (az > max(ax, ay)):
        _idx = 4
        c = z 
    elif (ay > ax):
        _idx = 2
        c = y
        y = z 
    else:
        _idx = 0
        c = x
        x = z 

    if (c < 0.0):
        _idx += 1

    m = wp.trunc(abs(c)) / 2
    ms[b] = m

    # m0 = __uint_as_float(__float_as_uint(_m) ^ ((0x21u >> idx) << 31));
    # x = x * m0 + .5;
    m1 = -m if idx != 2 else m
    y = y * m1 + .5;
    ys[b] = y
    

def compute_adjacent_faces(xyz, w, cache):
    N = xyz.shape[0]
    device = xyz.device
    adj = torch.zeros((N, 6), device=device)
    # TODO
    return adj

class IntegralCubemap(torch.nn.Module):
    def __init__(self, init_val, bg_resolution):
        data = init_val * torch.ones((1, 6, bg_resolution, bg_resolution, 3))
        self.bg_mat = nn.Parameter(data)
        cache = tabularize_adjacencies()
        self.register_buffer('cache', cache)

    def sa2mip(self, u, saSample, eps=torch.finfo(torch.float32).eps):
        h, w = self.bg_mat.shape[-2], self.bg_mat.shape[-3]
        saTexel = 4 * math.pi / (6*h*w) * 4
        # TODO calculate distortion of cube map for saTexel
        # distortion = 4 * math.pi / 6
        distortion = 1/torch.linalg.norm(u, dim=-1, ord=torch.inf).reshape(*saSample.shape)
        saTexel = distortion / h / w
        # saTexel is the ratio to the solid angle subtended by one pixel of the 0th mipmap level
        # num_pixels = self.bg_mat.numel() // 3
        # saTexel = distortion / num_pixels
        miplevel = ((saSample - torch.log(saTexel.clip(min=eps))) / math.log(2))/2 + self.mipbias + self.mipnoise * torch.rand_like(saSample)
        
        return miplevel.clip(0)#, max=int(math.log(h) / math.log(2))-2)

    def forward(self, viewdirs, saSample, max_level=None):
        miplevel = self.sa2mip(viewdirs, saSample)
        w = 2 ** miplevel
        adj = compute_adjacent_faces(viewdirs, w, self.cache)
