import math
import torch
from models import safemath
from scipy.special import legendre as legendrecoeffs
import numpy as np
from icecream import ic
from .sh import eval_sh_bases

def legendre(l, x):
    c = torch.tensor(legendrecoeffs(l).c[::-1].copy(), device=x.device, dtype=torch.float32)
    xpow = x[..., None]**torch.arange(len(c), device=x.device)
    return (xpow * c).sum(dim=-1)

def Yl(theta, phi, l):
    val = (-1)**l/2**l/math.factorial(l)*math.sqrt(math.factorial(2*l+1)/4/math.pi)*torch.sin(theta)**l*torch.exp(1j*l*phi)
    return val.real, val.imag

def Y0(theta, l):
    v = legendre(l, torch.cos(theta))
    return math.sqrt((2*l+1)/4/math.pi)*v

def Al(l, kappa):
    return torch.exp(-l*(l+1)/2/kappa)

class FullISH(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        self.max_degree = max_degree

    def dim(self):
        return (self.max_degree+1)**2

    def forward(self, vecs, kappa):
        base = eval_sh_bases(self.max_degree, vecs)
        return base

class ISH(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        self.max_degree = max_degree

    def dim(self):
        return 4*self.max_degree
        
    def forward(self, vec, kappa):
        a, b, c = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
        norm2d = torch.sqrt(a**2+b**2)
        phi = safemath.atan2(b, a)
        theta = safemath.atan2(c, norm2d) - np.pi/2
        
        degrees = range(1, self.max_degree+1)
        Als = [Al(l, kappa)[..., None] for l in degrees]
        # vert = [Als[l]*Y0(theta, 2**l) for l in range(self.max_degree)]
        # horz = sum([[Als[l]*v for v in Yl(theta, phi, 2**l)] for l in range(self.max_degree)], [])
        vert1 = torch.stack([Als[l-1]*Y0(theta, l) for l in degrees], dim=2)
        vert2 = torch.stack([Als[l-1]*Y0(theta, l-1) for l in degrees], dim=2)
        horz = torch.stack(sum([[Als[l-1]*v for v in Yl(theta, phi, l-1)] for l in degrees], []), dim=1).reshape(-1, self.max_degree, 2).permute(0, 2, 1)
        return torch.cat([vert1, vert2, horz], dim=1)


class RandISH(torch.nn.Module):
    def __init__(self, max_degree, rand_n, std=10):
        super().__init__()
        self.ish = ISH(max_degree)
        self.max_degree = max_degree
        self.rand_n = rand_n
        matrices = torch.normal(0, std, (rand_n, 3, 3))
        self.register_buffer('matrices', matrices)
        
    def dim(self):
        return self.rand_n * self.ish.dim()
    
    def forward(self, vec, kappa):
        outs = []
        for mat in self.matrices:
            outs.append(self.ish(vec @ mat, kappa))
        return torch.cat(outs, dim=1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    res = 40
    ele_grid, azi_grid = torch.meshgrid(
        torch.linspace(-np.pi/2, np.pi/2, res, dtype=torch.float32),
        torch.linspace(-np.pi, np.pi, 2*res, dtype=torch.float32), indexing='ij')

    azis = azi_grid.reshape(-1)
    eles = ele_grid.reshape(-1)
# each col of x ranges from -pi/2 to pi/2
# each row of y ranges from -pi to pi
    ang_vecs = torch.stack([
        torch.cos(eles) * torch.cos(azis),
        torch.cos(eles) * torch.sin(azis),
        -torch.sin(eles),
    ], dim=-1).reshape(-1, 3)
    # ang_vecs.requires_grad = True
    
    max_deg = 5
    ise = ISH(max_deg)
    coeffs = ise(ang_vecs, 20*torch.ones(ang_vecs.shape[0]))
    ic(coeffs.shape)

    for deg in range(max_deg):
        fig, ax = plt.subplots(4)
        ax[0].imshow(coeffs[:, 0, deg].reshape(res, 2*res))
        ax[1].imshow(coeffs[:, 1, deg].reshape(res, 2*res))
        ax[2].imshow(coeffs[:, 2, deg].reshape(res, 2*res))
        ax[3].imshow(coeffs[:, 3, deg].reshape(res, 2*res))
        # ax[1, 1].imshow(coeffs[:, 3, deg].reshape(res, 2*res))

    plt.show()