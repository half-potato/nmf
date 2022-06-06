from sympy import *
from icecream import ic
import sympy
import numpy as np
import matplotlib.pyplot as plt
import torch
import functools
import operator
import math
from sympy.parsing import mathematica, latex
import re

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
    print(poly)
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
kappa_v = 1/np.sqrt(1/180*np.pi)

# def integral(a, b, c, k, m):
#     numer = 4*a*k*m*(8*a**4*k**4*(149-30*m**2+m**4)-10*a**2*k**2*(16**2*k**2*(-13+m^2)-(1+2*c**2*k**2)*(-5+m)*(5+m)*(m**2-7))+15*(64*b**4*k**4-4*b**2*k**2*(1+2*c**2*k**2)*(m**2-25)+(1+c**2*k**2+c**4*k**4)*(255-34*m**2+m**4)))*math.sin(m*np.pi)
#     denom = 15*(m-5)*(m-3)*(m-1)*(m+1)*(m+3)*(m+5)
#     return -numer / denom
# eq = mathematica.mathematica("-((a k m (1024 a^12 k^12 (8446433085-2392136682 m^2+177934471 m^4-5281276 m^6+71643 m^8-442 m^10+m^12)+3003 (2123366400 b^12 k^12-88473600 b^10 k^10 (22+c^2 k^2) (-169+m^2)+2211840 b^8 k^8 (792+44 c^2 k^2+c^4 k^4) (20449-290 m^2+m^4)-46080 b^6 k^6 (33264+2376 c^2 k^2+66 c^4 k^4+c^6 k^6) (-1656369+43939 m^2-371 m^4+m^6)+960 b^4 k^4 (1330560+133056 c^2 k^2+4752 c^4 k^4+88 c^6 k^6+c^8 k^8) (81162081-3809380 m^2+62118 m^4-420 m^6+m^8)-24 b^2 k^2 (39916800+6652800 c^2 k^2+332640 c^4 k^4+7920 c^6 k^6+110 c^8 k^8+c^10 k^10) (-2029052025+176396581 m^2-5362330 m^4+72618 m^6-445 m^8+m^10)+(479001600+239500800 c^2 k^2+19958400 c^4 k^4+665280 c^6 k^6+11880 c^8 k^8+132 c^10 k^10+c^12 k^12) (18261468225-3616621254 m^2+224657551 m^4-6015892 m^6+76623 m^8-454 m^10+m^12))-12012 a^2 k^2 (88473600 b^10 k^10 (-37+m^2)-3686400 b^8 k^8 (22+c^2 k^2) (5239-200 m^2+m^4)+92160 b^6 k^6 (792+44 c^2 k^2+c^4 k^4) (-511225+27699 m^2-315 m^4+m^6)-1920 b^4 k^4 (33264+2376 c^2 k^2+66 c^4 k^4+c^6 k^6) (31471011-2491210 m^2+50988 m^4-390 m^6+m^8)+40 b^2 k^2 (1330560+133056 c^2 k^2+4752 c^4 k^4+88 c^6 k^6+c^8 k^8) (-1055107053+130684021 m^2-4616914 m^4+67578 m^6-433 m^8+m^10)-(39916800+6652800 c^2 k^2+332640 c^4 k^4+7920 c^6 k^6+110 c^8 k^8+c^10 k^10) (14203364175-3263828092 m^2+213932891 m^4-5870656 m^6+75733 m^8-452 m^10+m^12))+24024 a^4 k^4 (2211840 b^8 k^8 (1909-110 m^2+m^4)-92160 b^6 k^6 (22+c^2 k^2) (-217841+16499 m^2-259 m^4+m^6)+2304 b^4 k^4 (792+44 c^2 k^2+c^4 k^4) (16134261-1660240 m^2+41538 m^4-360 m^6+m^8)-48 b^2 k^2 (33264+2376 c^2 k^2+66 c^4 k^4+c^6 k^6) (-677454921+100789501 m^2-4005058 m^4+62898 m^6-421 m^8+m^10)+(1330560+133056 c^2 k^2+4752 c^4 k^4+88 c^6 k^6+c^8 k^8) (12093150069-3002460050 m^2+204699063 m^4-5735500 m^6+74867 m^8-450 m^10+m^12))-27456 a^6 k^6 (46080 b^6 k^6 (-110937+10339 m^2-203 m^4+m^6)-1920 b^4 k^4 (22+c^2 k^2) (9599031-1165270 m^2+33768 m^4-330 m^6+m^8)+48 b^2 k^2 (792+44 c^2 k^2+c^4 k^4) (-483843789+80866621 m^2-3506602 m^4+58578 m^6-409 m^8+m^10)-(33264+2376 c^2 k^2+66 c^4 k^4+c^6 k^6) (10738240227-2800881048 m^2+196688947 m^4-5609704 m^6+74025 m^8-448 m^10+m^12))+18304 a^8 k^8 (960 b^4 k^4 (6270921-855100 m^2+27678 m^4-300 m^6+m^8)-40 b^2 k^2 (22+c^2 k^2) (-368655417+66883381 m^2-3101386 m^4+54618 m^6-397 m^8+m^10)+(792+44 c^2 k^2+c^4 k^4) (9770552649-2639147806 m^2+189675743 m^4-5492548 m^6+73207 m^8-446 m^10+m^12))-6656 a^10 k^10 (24 b^2 k^2 (-293404365+56622181 m^2-2769250 m^4+51018 m^6-385 m^8+m^10)-(22+c^2 k^2) (9033241815-2505381044 m^2+183472971 m^4-5383312 m^6+72413 m^8-444 m^10+m^12))) Sin[m Pi])/(359610451200 (-13+m) (-11+m) (-9+m) (-7+m) (-5+m) (-3+m) (-1+m) (1+m) (3+m) (5+m) (7+m) (9+m) (11+m) (13+m)))")

def mparse(s):
    s = re.sub('\.([^0-9])', r'\g<1>', s)
    s = s.replace('\\[Pi]', 'Pi')
    eq = mathematica.mathematica(s, {'Csch[x]': '1/Sinh[x]'})
    return eq
eqs = []
with open('eq.text', 'r') as f:
    for s in f.readlines():
        integral = lambdify(['a', 'b', 'c', 'k'], mparse(s))
        eqs.append(integral)

s = "1.27728*10^-16 b k^2 (1.6614*10^15-3.3228*10^14 a^2 k^2+6.6456*10^14 b^2 k^2+8.307*10^14 c^2 k^2-4.74686*10^13 a^4 k^4+2.10971*10^13 a^2 b^2 k^4+4.21943*10^13 b^4 k^4-5.538*10^13 a^2 c^2 k^4+1.1076*10^14 b^2 c^2 k^4+6.9225*10^13 c^4 k^4-1.7581*10^12 a^6 k^6-7.03238*10^11 a^4 b^2 k^6+1.68777*10^12 a^2 b^4 k^6+1.12518*10^12 b^6 k^6-4.74686*10^12 a^4 c^2 k^6+2.10971*10^12 a^2 b^2 c^2 k^6+4.21943*10^12 b^4 c^2 k^6-2.769*10^12 a^2 c^4 k^6+5.538*10^12 b^2 c^4 k^6+2.3075*10^12 c^6 k^6-3.19654*10^10 a^8 k^8-3.65319*10^10 a^6 b^2 k^8+1.46127*10^10 a^4 b^4 k^8+4.17507*10^10 a^2 b^6 k^8+1.67003*10^10 b^8 k^8-1.25578*10^11 a^6 c^2 k^8-5.02313*10^10 a^4 b^2 c^2 k^8+1.20555*10^11 a^2 b^4 c^2 k^8+8.03701*10^10 b^6 c^2 k^8-1.69531*10^11 a^4 c^4 k^8+7.5347*10^10 a^2 b^2 c^4 k^8+1.50694*10^11 b^4 c^4 k^8-6.59286*10^10 a^2 c^6 k^8+1.31857*10^11 b^2 c^6 k^8+4.12054*10^10 c^8 k^8-3.51268*10^8 a^10 k^10-6.50496*10^8 a^8 b^2 k^10-2.08159*10^8 a^6 b^4 k^10+5.35265*10^8 a^4 b^6 k^10+5.5509*10^8 a^2 b^8 k^10+1.58597*10^8 b^10 k^10-1.77585*10^9 a^8 c^2 k^10-2.02955*10^9 a^6 b^2 c^2 k^10+8.11819*10^8 a^4 b^4 c^2 k^10+2.31948*10^9 a^2 b^6 c^2 k^10+9.27793*10^8 b^8 c^2 k^10-3.48828*10^9 a^6 c^4 k^10-1.39531*10^9 a^4 b^2 c^4 k^10+3.34875*10^9 a^2 b^4 c^4 k^10+2.2325*10^9 b^6 c^4 k^10-3.13946*10^9 a^4 c^6 k^10+1.39531*10^9 a^2 b^2 c^6 k^10+2.79063*10^9 b^4 c^6 k^10-9.15675*10^8 a^2 c^8 k^10+1.83135*10^9 b^2 c^8 k^10+4.57837*10^8 c^10 k^10-2.60198*10^6 a^12 k^12-6.62323*10^6 a^10 b^2 k^12-5.67706*10^6 a^8 b^4 k^12+2.16269*10^6 a^6 b^6 k^12+7.20896*10^6 a^4 b^8 k^12+4.71859*10^6 a^2 b^10 k^12+1.04858*10^6 b^12 k^12-1.59667*10^7 a^10 c^2 k^12-2.9568*10^7 a^8 b^2 c^2 k^12-9.46176*10^6 a^6 b^4 c^2 k^12+2.43302*10^7 a^4 b^6 c^2 k^12+2.52314*10^7 a^2 b^8 c^2 k^12+7.20896*10^6 b^10 c^2 k^12-4.03603*10^7 a^8 c^4 k^12-4.61261*10^7 a^6 b^2 c^4 k^12+1.84504*10^7 a^4 b^4 c^4 k^12+5.27155*10^7 a^2 b^6 c^4 k^12+2.10862*10^7 b^8 c^4 k^12-5.28528*10^7 a^6 c^6 k^12-2.11411*10^7 a^4 b^2 c^6 k^12+5.07387*10^7 a^2 b^4 c^6 k^12+3.38258*10^7 b^6 c^6 k^12-3.56756*10^7 a^4 c^8 k^12+1.58558*10^7 a^2 b^2 c^8 k^12+3.17117*10^7 b^4 c^8 k^12-8.32432*10^6 a^2 c^10 k^12+1.66486*10^7 b^2 c^10 k^12+3.46847*10^6 c^12 k^12) Csch[k]"
integral = lambdify(['a', 'b', 'c', 'k', 'm'], mparse(s))

# eq = latex.parse_latex('-\frac{\text{2.7807868115708424$\grave{ }$*${}^{\wedge}$-12} a k m \left(1024. a^{12} \left(m^{12}-442. m^{10}+71643. m^8-5.28128\times 10^6 m^6+1.77934\times 10^8 m^4-2.39214\times 10^9 m^2+8.44643\times 10^9\right) k^{12}-6656. a^{10} \left(24. b^2 k^2 \left(m^{10}-385. m^8+51018. m^6-2.76925\times 10^6 m^4+5.66222\times 10^7 m^2-2.93404\times 10^8\right)-1. \left(c^2 k^2+22.\right) \left(m^{12}-444. m^{10}+72413. m^8-5.38331\times 10^6 m^6+1.83473\times 10^8 m^4-2.50538\times 10^9 m^2+9.03324\times 10^9\right)\right) k^{10}+18304. a^8 \left(960. b^4 \left(m^8-300. m^6+27678. m^4-855100. m^2+6.27092\times 10^6\right) k^4-40. b^2 \left(c^2 k^2+22.\right) \left(m^{10}-397. m^8+54618. m^6-3.10139\times 10^6 m^4+6.68834\times 10^7 m^2-3.68655\times 10^8\right) k^2+\left(c^4 k^4+44. c^2 k^2+792.\right) \left(m^{12}-446. m^{10}+73207. m^8-5.49255\times 10^6 m^6+1.89676\times 10^8 m^4-2.63915\times 10^9 m^2+9.77055\times 10^9\right)\right) k^8-27456. a^6 \left(46080. b^6 \left(m^6-203. m^4+10339. m^2-110937.\right) k^6-1920. b^4 \left(c^2 k^2+22.\right) \left(m^8-330. m^6+33768. m^4-1.16527\times 10^6 m^2+9.59903\times 10^6\right) k^4+48. b^2 \left(c^4 k^4+44. c^2 k^2+792.\right) \left(m^{10}-409. m^8+58578. m^6-3.5066\times 10^6 m^4+8.08666\times 10^7 m^2-4.83844\times 10^8\right) k^2-1. \left(c^6 k^6+66. c^4 k^4+2376. c^2 k^2+33264.\right) \left(m^{12}-448. m^{10}+74025. m^8-5.6097\times 10^6 m^6+1.96689\times 10^8 m^4-2.80088\times 10^9 m^2+1.07382\times 10^{10}\right)\right) k^6+24024. a^4 \left(2.21184\times 10^6 b^8 \left(m^4-110. m^2+1909.\right) k^8-92160. b^6 \left(c^2 k^2+22.\right) \left(m^6-259. m^4+16499. m^2-217841.\right) k^6+2304. b^4 \left(c^4 k^4+44. c^2 k^2+792.\right) \left(m^8-360. m^6+41538. m^4-1.66024\times 10^6 m^2+1.61343\times 10^7\right) k^4-48. b^2 \left(c^6 k^6+66. c^4 k^4+2376. c^2 k^2+33264.\right) \left(m^{10}-421. m^8+62898. m^6-4.00506\times 10^6 m^4+1.0079\times 10^8 m^2-6.77455\times 10^8\right) k^2+\left(c^8 k^8+88. c^6 k^6+4752. c^4 k^4+133056. c^2 k^2+1.33056\times 10^6\right) \left(m^{12}-450. m^{10}+74867. m^8-5.7355\times 10^6 m^6+2.04699\times 10^8 m^4-3.00246\times 10^9 m^2+1.20932\times 10^{10}\right)\right) k^4-12012. a^2 \left(8.84736\times 10^7 b^{10} \left(m^2-37.\right) k^{10}-3.6864\times 10^6 b^8 \left(c^2 k^2+22.\right) \left(m^4-200. m^2+5239.\right) k^8+92160. b^6 \left(c^4 k^4+44. c^2 k^2+792.\right) \left(m^6-315. m^4+27699. m^2-511225.\right) k^6-1920. b^4 \left(c^6 k^6+66. c^4 k^4+2376. c^2 k^2+33264.\right) \left(m^8-390. m^6+50988. m^4-2.49121\times 10^6 m^2+3.1471\times 10^7\right) k^4+40. b^2 \left(c^8 k^8+88. c^6 k^6+4752. c^4 k^4+133056. c^2 k^2+1.33056\times 10^6\right) \left(m^{10}-433. m^8+67578. m^6-4.61691\times 10^6 m^4+1.30684\times 10^8 m^2-1.05511\times 10^9\right) k^2-1. \left(c^{10} k^{10}+110. c^8 k^8+7920. c^6 k^6+332640. c^4 k^4+6.6528\times 10^6 c^2 k^2+3.99168\times 10^7\right) \left(m^{12}-452. m^{10}+75733. m^8-5.87066\times 10^6 m^6+2.13933\times 10^8 m^4-3.26383\times 10^9 m^2+1.42034\times 10^{10}\right)\right) k^2+3003. \left(2.12337\times 10^9 b^{12} k^{12}-8.84736\times 10^7 b^{10} \left(c^2 k^2+22.\right) \left(m^2-169.\right) k^{10}+2.21184\times 10^6 b^8 \left(c^4 k^4+44. c^2 k^2+792.\right) \left(m^4-290. m^2+20449.\right) k^8-46080. b^6 \left(c^6 k^6+66. c^4 k^4+2376. c^2 k^2+33264.\right) \left(m^6-371. m^4+43939. m^2-1.65637\times 10^6\right) k^6+960. b^4 \left(c^8 k^8+88. c^6 k^6+4752. c^4 k^4+133056. c^2 k^2+1.33056\times 10^6\right) \left(m^8-420. m^6+62118. m^4-3.80938\times 10^6 m^2+8.11621\times 10^7\right) k^4-24. b^2 \left(c^{10} k^{10}+110. c^8 k^8+7920. c^6 k^6+332640. c^4 k^4+6.6528\times 10^6 c^2 k^2+3.99168\times 10^7\right) \left(m^{10}-445. m^8+72618. m^6-5.36233\times 10^6 m^4+1.76397\times 10^8 m^2-2.02905\times 10^9\right) k^2+\left(c^{12} k^{12}+132. c^{10} k^{10}+11880. c^8 k^8+665280. c^6 k^6+1.99584\times 10^7 c^4 k^4+2.39501\times 10^8 c^2 k^2+4.79002\times 10^8\right) \left(m^{12}-454. m^{10}+76623. m^8-6.01589\times 10^6 m^6+2.24658\times 10^8 m^4-3.61662\times 10^9 m^2+1.82615\times 10^{10}\right)\right)\right) \sin (3.14159 m)}{(m-13.) (m-11.) (m-9.) (m-7.) (m-5.) (m-3.) (m-1.) (m+1.) (m+3.) (m+5.) (m+7.) (m+9.) (m+11.) (m+13.)}')

for deg in range(1, 10):
    ic(kappa_v, deg)
    # m = 2**deg
    m = deg
    
    # signs = []
    # vals = []
    # for coeff, power in C[deg]:
    #     sig = float(sign(coeff))
    #     lcoeff = float(log(sig*coeff))
    #     p = ipes[power](kappa_v*c, kappa_v*torch.sqrt(a**2+b**2))

    #     # plt.imshow(p.reshape(res, 2*res))
    #     # plt.show()
    #     # s += coeff * p
    #     signs.append(sig)
    #     vals.append(p+lcoeff)
    # vals = torch.stack(vals, dim=0)
    # signs = torch.tensor(signs).reshape(-1, 1)
    # maxval = vals.max(dim=0, keepdim=True).values
    # ic(maxval.shape, vals.shape)
    # ls = (((vals - maxval).exp() * signs).sum(dim=0) + maxval)
    # s = ls * kappa_v / (2*np.pi * math.sinh(kappa_v))
    # ic(s.shape)

    # s = eqs[m-1](a.double(), b.double(), c.double(), kappa_v.astype(np.float64))
    s = integral(a.double(), b.double(), c.double(), kappa_v.astype(np.float64), -m)

    plt.imshow(s.reshape(res, 2*res))
    plt.show()