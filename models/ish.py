import math
import torch
from models import safemath
from scipy.special import legendre as legendrecoeffs
import numpy as np
from icecream import ic
from scipy.spatial.transform import Rotation
from .sh import eval_sh_bases, eval_sh_bases_scaled, sh_basis
import functools
import operator

def legendre(l, x):
    c = torch.tensor(legendrecoeffs(l).c[::-1].copy(), device=x.device, dtype=torch.float32)
    xpow = x[..., None]**torch.arange(len(c), device=x.device)
    return (xpow * c).sum(dim=-1)

def Yl(theta, phi, l):
    # coeff = (-1)**l/2**l/math.factorial(l)*math.sqrt(math.factorial(2*l+1)/4/math.pi)
    logcoeff = -2*math.log(l) - math.lgamma(l+1) + 0.5 * (math.lgamma(2*l+2) - math.log(4*math.pi))
    val = (-1)**l * math.exp(logcoeff) * torch.sin(theta)**l*torch.exp(1j*l*phi)
    return val.real, val.imag

def Y0(theta, l):
    v = legendre(l, torch.cos(theta))
    return math.sqrt((2*l+1)/4/math.pi)*v

def Al(l, kappa):
    return torch.exp(-l*(l+1)/2/(kappa+1e-8))

def rising_factorial(z, m):
    if m == 0:
        return 1
    if z < 0 and z % 1 == 0:
        return 0
    return math.gamma(z+m) / math.gamma(z)

class SHBasis(torch.nn.Module):
    def __init__(self, deg):
        super(SHBasis, self).__init__()
        self.deg = deg.item()
        # assert(self.deg < 7)
        l = deg
        coeffs = torch.tensor(legendrecoeffs(l).c[::-1].copy(), dtype=torch.float32)
        logcoeff = -2*math.log(l) - math.lgamma(l+1) + 0.5 * (math.lgamma(2*l+2) - math.log(4*math.pi))
        self.coeff = (-1)**self.deg * math.exp(logcoeff)
        self.register_buffer('coeffs', coeffs)

    def forward(self, theta, phi, kappa):
        a = Al(self.deg, kappa)

        # calculate legendre of cos of theta
        x = torch.cos(theta)
        xpow = x[..., None]**torch.arange(len(self.coeffs), device=x.device)
        v = (xpow * self.coeffs).sum(dim=-1)
        y0 = math.sqrt((2*self.deg+1)/4/math.pi)*v

        yl =  self.coeff * torch.sin(theta)**self.deg*torch.exp(1j*self.deg*phi)
        yl1, yl2 = yl.real, yl.imag

        return a*torch.cat([y0, yl1, yl2], dim=-1)

class LHyperGeom(torch.nn.Module):
    def __init__(self, upper, lower, N=20):
        super().__init__()
        upper_coeffs = torch.tensor([
            # sum([rising_factorial(n, k) for n in upper])
            functools.reduce(operator.mul, [rising_factorial(a, k) for a in upper], 1) / math.factorial(k)
            for k in range(N)
        ])
        lower_coeffs = torch.tensor([
            # sum([rising_factorial(n, k) for n in lower]) + math.lgamma(k+1)
            functools.reduce(operator.mul, [rising_factorial(a, k) for a in lower], 1)
            for k in range(N)
        ])
        self.register_buffer('upper_coeffs', upper_coeffs.unsqueeze(0))
        self.register_buffer('lower_coeffs', lower_coeffs.unsqueeze(0))
        self.N = N
        
    def forward(self, x):
        # sgn = torch.sign(x).unsqueeze(-1)**torch.arange(self.N)
        # lx = torch.log(torch.abs(x)).unsqueeze(-1)*torch.arange(self.N).reshape(1, -1)
        # lx = lx + self.upper_coeffs - self.lower_coeffs
        # # lx = lx + self.upper_coeffs.log() - self.lower_coeffs.log()
        # maxval = torch.max(lx, dim=-1).values
        # s = ((lx - maxval.unsqueeze(-1)).exp()*sgn).sum(dim=-1).log() + maxval

        expx = x.unsqueeze(-1)**torch.arange(self.N)
        # ic(lx, expx)
        s2 = (self.upper_coeffs * expx / self.lower_coeffs)
        # s2 = (self.upper_coeffs.exp() * expx / self.lower_coeffs.exp())
        # ic(s2.sum(dim=-1))
        return s2.sum(dim=-1)

class ListISH(torch.nn.Module):
    def __init__(self, degs=[0,1,2,4,8,16]):
        super().__init__()
        self.degs = degs

    def dim(self):
        return sum([2*deg+1 for deg in self.degs])

    def forward(self, vecs, roughness):
        kappa = 1/(roughness+1e-3)
        base = sh_basis(self.degs, vecs, kappa)
        return base

class FullISH(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        self.max_degree = max_degree

    def dim(self):
        return (self.max_degree+1)**2

    def forward(self, vecs, roughness):
        kappa = 1/(roughness+1e-8)
        base = eval_sh_bases_scaled(self.max_degree, vecs, kappa)
        return base

def compute_ortho_basis(phi, theta, kappa, deg):
    Als = Al(deg, kappa)[..., None]
    vert1 = Y0(theta, deg)
    vert2 = Y0(theta, deg-1)
    horz1, horz2 = Yl(theta, phi, deg-1)
    return Als*torch.cat([vert1, vert2, horz1, horz2], dim=1)

class ISH(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        self.max_degree = max_degree
        self.degrees = 2**torch.arange(0, self.max_degree)
        # degrees = torch.arange(2, self.max_degree+1)
        self.basii = torch.nn.ModuleList([SHBasis(deg) for deg in self.degrees])

    def dim(self):
        return 3*len(self.degrees)
    
    def forward(self, vec, roughness):
        kappa = 1/(roughness+1e-8)
        a, b, c = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
        norm2d = torch.sqrt(a**2+b**2)
        phi = safemath.atan2(b, a)
        theta = safemath.atan2(c, norm2d) - np.pi/2
        output = torch.cat([basis(theta, phi, kappa) for basis in self.basii], dim=1)
        return output
        
        # Als = [Al(l, kappa)[..., None] for l in self.degrees]
        # vert1 = torch.stack([Als[i]*Y0(theta, l) for i, l in enumerate(self.degrees)], dim=2)
        # vert2 = torch.stack([Als[i]*Y0(theta, l-1) for i, l in enumerate(self.degrees)], dim=2)
        # horz = torch.stack(sum([[Als[i]*v for v in Yl(theta, phi, l-1)] for i, l in enumerate(self.degrees)], []), dim=1).reshape(-1, len(self.degrees), 2).permute(0, 2, 1)
        # return torch.cat([vert1, vert2, horz], dim=1)

class FractionalY0(torch.nn.Module):
    def __init__(self, degree, N=20) -> None:
        super().__init__()
        self.hypergeom = LHyperGeom([-degree, degree+1], [1], N=N)
        self.degree = degree
        
    def forward(self, theta):
        return math.sqrt((2*self.degree+1)/4/math.pi)*self.hypergeom((1-torch.cos(theta))/2)

class RandRotISH(torch.nn.Module):
    def __init__(self, rand_n, core_degs=[1,2,4,8], rand_degs=[16]):
        super().__init__()
        self.rand_n = rand_n
        angs = torch.rand(rand_n, 3)*2*np.pi
        matrices = []
        for ang in angs:
            matrices.append(torch.as_tensor(Rotation.from_euler('xyz', ang).as_matrix()))
        matrices = torch.stack(matrices, dim=0).float()
        self.core_basis = ListISH(core_degs)
        self.rand_basis = ListISH(rand_degs)
        self.register_buffer('matrices', matrices)
        
    def dim(self):
        return self.rand_n*self.rand_basis.dim() + self.core_basis.dim()
    
    def forward(self, vec, roughness):
        B = vec.shape[0]

        evec = vec.reshape(B, 1, 3).expand(B, self.rand_n, 3).reshape(-1, 1, 3)
        emats = self.matrices.reshape(1, -1, 3, 3).expand(B, self.rand_n, 3, 3).reshape(-1, 3, 3)
        eroughness = roughness.reshape(B, 1, 1).expand(B, self.rand_n, 1).reshape(-1, 1)
        rvecs = torch.matmul(evec, emats).reshape(-1, 3)
        rbasis = self.rand_basis(rvecs, eroughness).reshape(B, -1)
        outbasis = torch.cat([self.core_basis(vec, roughness), rbasis], dim=-1)
        return outbasis

class RandISH(torch.nn.Module):
    def __init__(self, rand_n, std=10):
        super().__init__()
        self.rand_n = rand_n
        # matrices = torch.normal(0, std, (rand_n, 3, 3))
        # matrices = torch.normal(0, std, (rand_n, 3, 3))
        # matrices = torch.normal(0, std, (rand_n, 2))
        # """
        angs = torch.rand(rand_n, 3)*2*np.pi
        matrices = []
        for ang in angs:
            matrices.append(torch.as_tensor(Rotation.from_euler('xyz', ang).as_matrix()))
        matrices = torch.stack(matrices, dim=0).float()
        # """
        degrees = torch.normal(0, std, (rand_n,1)).clip(min=1, max=9).int()
        self.basii = torch.nn.ModuleList([SHBasis(deg) for deg in degrees])
        # self.y0s = [FractionalY0(degree[0]) for degree in degrees]
        self.register_buffer('matrices', matrices)
        self.register_buffer('degrees', degrees)
        
    def dim(self):
        return self.rand_n*2
    
    def forward(self, vec, roughness):
        B = vec.shape[0]
        kappa = 1/(roughness+1e-8)

        a, b, c = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
        norm2d = torch.sqrt(a**2+b**2)
        phi = safemath.atan2(b, a)
        theta = safemath.atan2(c, norm2d) - np.pi/2

        outs = []

        # t_theta = theta.reshape(-1, 1, 1).expand(B, self.rand_n, 1)
        # t_phi = (self.matrices[None, :, 0] * theta + self.matrices[None, :, 1] * phi).reshape(B, self.rand_n, 1)
        #
        # x = torch.cos(theta)
        # xpow = x[..., None]**torch.arange(len(self.coeffs), device=x.device)
        # v = (xpow * self.coeffs).sum(dim=-1)
        # y0 = math.sqrt((2*self.deg+1)/4/math.pi)*v
        #
        # yl =  self.coeff * torch.sin(theta)**self.deg*torch.exp(1j*self.deg*phi)
        # yl1, yl2 = yl.real, yl.imag

        for i, (basis, deg, mat) in enumerate(zip(self.basii, self.degrees, self.matrices)):
        #  for i, (deg, mat) in enumerate(zip(self.degrees, self.matrices)):
            # ob = compute_ortho_basis(phi, theta, kappa, deg)
            rvec = vec @ mat
            a, b, c = rvec[:, 0:1], rvec[:, 1:2], rvec[:, 2:3]
            norm2d = torch.sqrt(a**2+b**2)
            t_phi = safemath.atan2(b, a)
            t_theta = safemath.atan2(c, norm2d) - np.pi/2

            # t_theta = theta
            # t_phi = mat[0] * theta + mat[1] * phi

            ind = 1 if deg > 0 else 2
            degs = basis(t_theta.reshape(-1, 1), t_phi.reshape(-1, 1), kappa.reshape(-1, 1))
            #  outs.append(out)
            out = torch.stack([degs[:, 0], degs[:, ind]], dim=1)
            # ic(out.max(dim=0).values, deg, kappa)
            outs.append(out)
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
    # ise = ISH(max_deg)
    ise = RandISH(10)
    coeffs = ise(ang_vecs, 20*torch.ones(ang_vecs.shape[0]))
    ic(coeffs.shape)

    # for deg in range(coeffs.shape[2]):
    #     fig, ax = plt.subplots(coeffs.shape[1])
    #     for i in range(coeffs.shape[1]):
    #         ax[i].imshow(coeffs[:, i, deg].reshape(res, 2*res))

    fig, ax = plt.subplots(coeffs.shape[1])
    for i in range(coeffs.shape[1]):
        ax[i].imshow(coeffs[:, i].reshape(res, 2*res))
        # ax[3].imshow(coeffs[:, 4, deg].reshape(res, 2*res))
        # ax[1, 1].imshow(coeffs[:, 3, deg].reshape(res, 2*res))

    plt.show()
