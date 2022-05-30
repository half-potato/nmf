from sympy import *
from icecream import ic
import sympy
import numpy as np
import matplotlib.pyplot as plt
import torch
import functools
import operator
import math

max_degree = 6

init_printing()
theta, phi = symbols('theta phi', real=True)
u0, u1, u2 = symbols('a b c', real=True)
u = Matrix([u0, u1, u2])
x0 = cos(theta) * cos(phi)
x1 = cos(theta) * sin(phi)
x2 = sin(theta)

kappa = symbols('kappa', positive=True)
n = symbols('n', positive=True, integer=True)

dp = x0 * u0 + x1 * u1 + x2 * u2
pdf = kappa * exp(kappa * dp)/(2 * pi * sinh(kappa))

def base_case(a, b):
    return 2*(a**2+b**2)**(-1/2)*sinh(sqrt(a**2+b**2))

def recursive_case(a, b, n):
    if n == 0:
        return base_case(a, b)
    else:
        if n % 2 == 0:
            add = 2*base_case(a, b)
        else:
            add = 0
        return add - n * recursive_case(a, b, n-1)
    
u = symbols('u')
C2 = [u, 2*u**2-1]
for i in range(2, 10):
    # C2.append(4*C2[i-1]*C2[i-2])
    C2.append(2*C2[i-1]**2 - 1)
C = []

for i in range(0, 10):
    s = expand(C2[i])
    coeffs = Poly(s, u).all_coeffs()
    poly = [(int(coeff), int(power)) for power, coeff in enumerate(coeffs) if coeff != 0]
    C.append(poly)

# test equations
res = 40
ele_grid, azi_grid = np.meshgrid(
    np.linspace(-np.pi/2, np.pi/2, res, dtype=np.float32),
    np.linspace(-np.pi, np.pi, 2*res, dtype=np.float32),
    indexing='ij')

azis = azi_grid.reshape(-1)
eles = ele_grid.reshape(-1)
# each col of x ranges from -pi/2 to pi/2
# each row of y ranges from -pi to pi
a = np.cos(eles) * np.cos(azis)
b = np.cos(eles) * np.sin(azis)
c = -np.sin(eles)
ang_vecs = np.stack([ a, b, c ], axis=-1).reshape(-1, 3)


def lrising_factorial(z, m):
    return math.lgamma(z+m) - math.lgamma(z)
    # if n == 0:
    #     return 0
    # return torch.log(torch.arange(start=0, end=n) + x).sum(dim=-1)

def rising_factorial(x, n):
    if n == 0:
        return 0
    return (torch.arange(start=0, end=n) + x).prod(dim=-1)


class LHyperGeom(torch.nn.Module):
    def __init__(self, upper, lower, N=20):
        super().__init__()
        upper_coeffs = torch.tensor([
            sum([lrising_factorial(n, k) for n in upper])
            # functools.reduce(operator.mul, [math.exp(lrising_factorial(a, k)) for a in upper], 1) / math.factorial(k)
            for k in range(N)
        ])
        lower_coeffs = torch.tensor([
            sum([lrising_factorial(n, k) for n in lower]) + math.lgamma(k+1)
            # functools.reduce(operator.mul, [math.exp(lrising_factorial(a, k)) for a in lower], 1)
            for k in range(N)
        ])
        self.register_buffer('upper_coeffs', upper_coeffs.unsqueeze(0))
        self.register_buffer('lower_coeffs', lower_coeffs.unsqueeze(0))
        self.N = N
        
    def forward(self, x):
        sgn = torch.sign(x).unsqueeze(-1)**torch.arange(self.N)
        lx = torch.log(torch.abs(x)).unsqueeze(-1)*torch.arange(self.N).reshape(1, -1)
        lx = lx + self.upper_coeffs - self.lower_coeffs
        # lx = lx + self.upper_coeffs.log() - self.lower_coeffs.log()
        maxval = torch.max(lx, dim=-1).values
        s = ((lx - maxval.unsqueeze(-1)).exp()*sgn).sum(dim=-1).log() + maxval

        # expx = x.unsqueeze(-1)**torch.arange(self.N)
        # ic(lx, expx)
        # # s2 = (self.upper_coeffs * expx / self.lower_coeffs)
        # s2 = (self.upper_coeffs.exp() * expx / self.lower_coeffs.exp())
        # ic(s2.sum(dim=-1))
        return s
    
# Test cases
# z = torch.tensor(0.5)
# a = 2
# b = 3
# ic(LHyperGeom(upper=[1, 1], lower=[2])(-z).exp())
# ic(torch.log(1+z)/z)
# ic(LHyperGeom(upper=[a, b], lower=[b])(z).exp())
# ic((1-z)**(-a))

class IPECosTheta(torch.nn.Module):
    def __init__(self, k, N=20):
        super().__init__()
        self.N = N
        coeffs = []
        self.geoms = []
        for m in range(N):
            upper = math.gamma(2*m+1/2) * k
            upper = -m * math.log(4) + math.lgamma(2*m+0.5) + math.lgamma(k+0.5)
            lower = math.lgamma(2*m+k+1) + 2*math.lgamma(m+1)
            coeff = upper - lower
            geom = LHyperGeom([1/2+k], [1/2, 2*m+k+1])
            self.geoms.append(geom)
            coeffs.append(coeff)
        self.coeffs = torch.tensor(coeffs).unsqueeze(0)
        self.k = k
        
    def forward(self, a, b):
        
        powb = torch.log(b).unsqueeze(-1)*torch.arange(self.N).reshape(1, -1)
        hgs = torch.stack([geom(a**2/4) for geom in self.geoms], dim=-1)
        ele = (hgs + powb + self.coeffs)
        s = torch.logsumexp(ele, dim=-1)
        return s

a = torch.as_tensor(a)
b = torch.as_tensor(b)
c = torch.as_tensor(c)

max_p = max([max(t[1] for t in p) for p in C])
ipes = [IPECosTheta(power) for power in range(max_p+1)]
kappa_v = 1/np.sqrt(0.1/180*np.pi)

for deg in range(8, 10):
    ic(kappa_v, deg)
    signs = []
    vals = []
    for coeff, power in C[deg]:
        sig = float(sign(coeff))
        lcoeff = float(log(sig*coeff))
        p = ipes[power](kappa_v*c, kappa_v*torch.sqrt(a**2+b**2))

        # plt.imshow(p.reshape(res, 2*res))
        # plt.show()
        # s += coeff * p
        signs.append(sig)
        vals.append(p+lcoeff)
    vals = torch.stack(vals, dim=0)
    signs = torch.tensor(signs).reshape(-1, 1)
    maxval = vals.max(dim=0, keepdim=True).values
    ic(maxval.shape, vals.shape)
    ls = (((vals - maxval).exp() * signs).sum(dim=0) + maxval)
    s = ls * kappa_v / (2*np.pi * math.sinh(kappa_v))
    ic(s.shape)


    plt.imshow(s.reshape(res, 2*res))
    plt.show()