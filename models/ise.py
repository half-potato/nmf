from pathlib import Path
import re
import numpy as np
import torch
from icecream import ic
import math

def read_coeffs(s):
    s = s.strip()
    s = re.sub('\.([^0-9])', r'\g<1>', s)
    s = s.replace('\\[Pi]', 'Pi') \
         .replace('{', '[') \
         .replace('}', ']') \
         .replace(' I', 'j') \
         .replace('^', '**')
    s = eval(s)
    mat = np.array(s)
    return torch.as_tensor(mat)
    

PHI_COEFFS = read_coeffs("""
{{{1., 0., 0., 0., 0.}, {0.5, 1., 0., 0., 0.}, {0.333333, 1., 1., 0., 
   0.}, {0.25, 1., 1.5, 1., 0.}, {0.2, 1., 2., 2., 1.}}, {{0.392699, 
   0., 0., 0., 0.}, {0.294524, 0.589049, 0., 0., 0.}, {0.230097, 
   0.736311, 0.736311, 0., 0.}, {0.187913, 0.80534, 1.28854, 0.859029,
    0.}, {0.158551, 0.845607, 1.81201, 1.93282, 
   0.966408}}, {{0.0736311, 0., 0., 0., 0.}, {0.0920388, 0.184078, 0.,
    0., 0.}, {0.0939563, 0.322136, 0.322136, 0., 0.}, {0.0906007, 
   0.422803, 0.724806, 0.483204, 0.}, {0.085646, 0.498304, 1.16271, 
   1.32881, 0.664405}}, {{0.00335558, 0., 0., 0., 0.}, {0.00755006, 
   0.0151001, 0., 0., 0.}, {0.0114195, 0.0415253, 0.0415253, 0., 
   0.}, {0.0146204, 0.0742265, 0.134957, 0.0899716, 0.}, {0.0171333, 
   0.109653, 0.27835, 0.337393, 0.168697}}, {{9.41388*10^-6, 0., 0., 
   0., 0.}, {0.000040009, 0.000080018, 0., 0., 0.}, {0.0001003, 
   0.000380085, 0.000380085, 0., 0.}, {0.000194002, 0.00105315, 
   0.00199545, 0.0013303, 0.}, {0.000320709, 0.00223102, 0.00605563, 
   0.00764922, 0.00382461}}, {{1.02368*10^-10, 0., 0., 0., 
   0.}, {8.44533*10^-10, 1.68907*10^-9, 0., 0., 0.}, {3.8035*10^-9, 
   1.47793*10^-8, 1.47793*10^-8, 0., 0.}, {1.23976*10^-8, 
   7.03648*10^-8, 1.36709*10^-7, 9.11392*10^-8, 0.}, {3.27374*10^-8, 
   2.41753*10^-7, 6.86057*10^-7, 8.88607*10^-7, 
   4.44304*10^-7}}, {{1.69194*10^-20, 0., 0., 0., 0.}, {2.7494*10^-19,
    5.49879*10^-19, 0., 0., 0.}, {2.33751*10^-18, 9.21047*10^-18, 
   9.21047*10^-18, 0., 0.}, {1.38419*10^-17, 8.0644*10^-17, 
   1.58881*10^-16, 1.0592*10^-16, 0.}, {6.41332*10^-17, 
   4.91387*10^-16, 1.43143*10^-15, 1.88009*10^-15, 9.40044*10^-16}}}
""")

THETA_COEFFS = read_coeffs("""
   {{{0. - 0.25 I, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0.106103, 0.333333, 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0. - 0.1875 I, 0. - 0.294524 I, 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0.}, {0.063662, 0.4, 0.2, 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0. - 0.15625 I, 0. - 0.490874 I, 0. - 0.184078 I, 0., 0., 0.,
    0., 0., 0., 0., 0., 0.}}, {{-0.125, 0.19635, 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0.}, {0. - 0.127324 I, 0. - 0.4 I, 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0.}, {-0.125, 0., 0.147262, 0., 0., 0., 0., 0.,
    0., 0., 0., 0.}, {0. - 0.0909457 I, 0. - 0.571429 I, 
   0. - 0.285714 I, 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {-0.117188, -0.184078, 0.138058, 0.115049, 0., 0., 0., 0., 0.,
    0., 0., 0.}}, {{0.03125, -0.294524, 0.0368155, 0., 0., 0., 0., 0.,
    0., 0., 0., 0.}, {0. + 0.101051 I, 0. + 0.126984 I, 
   0. - 0.190476 I, 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0.046875, -0.368155, -0.276117, 0.0460194, 0., 0., 0., 0., 
   0., 0., 0., 0.}, {0. + 0.090027 I, 0. + 0.363636 I, 
   0. - 0.121212 I, 0. - 0.20202 I, 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0.0546875, -0.343612, -0.644272, -0.214757, 0.0469782, 0., 
   0., 0., 0., 0., 0., 0.}}, {{0.00195313, -0.0859029, 
   0.161068, -0.0536893, 0.00167779, 0., 0., 0., 0., 0., 0., 
   0.}, {0. + 0.0496516 I, 0. - 0.144796 I, 0. - 0.106623 I, 
   0. + 0.171123 I, 0. - 0.0230358 I, 0., 0., 0., 0., 0., 0., 
   0.}, {0.00488281, -0.207087, 0.241602, 0.201335, -0.113251, 
   0.00377503, 0., 0., 0., 0., 0., 0.}, {0. + 0.0511439 I, 
   0. - 0.12892 I, 0. - 0.415337 I, 0. + 0.158768 I, 0. + 0.244503 I, 
   0. - 0.0400095 I, 0., 0., 0., 0., 0., 0.}, {0.00805664, -0.329039, 
   0.142373, 0.664405, 0.103813, -0.161949, 0.00570973, 0., 0., 0., 
   0., 0.}}, {{7.62939*10^-6, -0.00143811, 0.0163585, -0.059981, 
   0.0843483, -0.0472351, 0.00984064, -0.000602488, 4.70694*10^-6, 0.,
    0., 0.}, {0. + 0.0216241 I, 0. - 0.0590613 I, 0. + 0.0833085 I, 
   0. - 0.0540137 I, 0. - 0.0789162 I, 0. + 0.126959 I, 
   0. - 0.0517717 I, 0. + 0.00649455 I, 0. - 0.000176482 I, 0., 0., 
   0.}, {0.0000343323, -0.00641755, 0.0687595, -0.20857, 0.143392, 
   0.129053, -0.150562, 0.0384086, -0.00252057, 0.0000200045, 0., 
   0.}, {0. + 0.0219284 I, 0. - 0.0817356 I, 0. + 0.127069 I, 
   0. + 0.0501927 I, 0. - 0.41098 I, 0. + 0.182185 I, 0. + 0.210434 I,
    0. - 0.135429 I, 0. + 0.0192567 I, 0. - 0.000542893 I, 0., 
   0.}, {0.0000905991, -0.0167929, 0.168747, -0.399187, -0.103199, 
   0.681113, -0.0851391, -0.26758, 0.0883698, -0.00622918, 
   0.0000501501, 0.}}}
""")

class ThetaISE(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        # pad coefficients to cube matrix
        self.max_degree = max_degree
        degrees = torch.arange(0, max_degree+1)
        freqs = 2**(degrees)
        # row = kappa degree, col = a^2 degree
        if max_degree > THETA_COEFFS.shape[0]:
            raise ValueError("max_degree must be less than {}".format(THETA_COEFFS.shape[0]))
        theta_coeffs = THETA_COEFFS[:self.max_degree+1]
        self.max_kappa_pow = theta_coeffs.shape[1]
        self.max_ab_pow = theta_coeffs.shape[2]

        ab_pow = torch.arange(self.max_ab_pow)
        kappa_pow = torch.arange(self.max_kappa_pow)

        j, k = torch.meshgrid(kappa_pow, ab_pow, indexing='ij')
        
        c_powers = ((j-2*k).int().unsqueeze(0) + freqs.reshape(-1, 1, 1) +1).clip(min=0)
        mul = torch.tensor([math.log(math.factorial(f)) for f in freqs]).reshape(1, -1)

        self.register_buffer('theta_coeffs', theta_coeffs.cfloat())
        self.register_buffer('mul', mul)
        self.register_buffer('c_powers', c_powers)
        self.register_buffer('ab_pow', ab_pow)
        self.register_buffer('kappa_pow', kappa_pow)
        self.register_buffer('freqs', freqs)

    def forward(self, vec, kappa):
        B = vec.shape[0]
        kappa = kappa.reshape(B, 1)
        a, b, c = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
        powab = (a**2+b**2).reshape(B, 1)**self.ab_pow
        powkappa = kappa**self.kappa_pow
        powc = c.reshape(B, 1, 1, 1)**self.c_powers
        # ic(self.c_power_offset)
        # ic((self.c_powers-self.c_power_offset).min())
        pow_matrix = powkappa.reshape(B, 1, -1, 1) * powab.reshape(B, 1, 1, -1) * powc

        sinh_norm = - (torch.sinh(kappa)+1e-8).log()
        mul = (kappa.log()*(self.freqs+1) - self.mul + sinh_norm).exp()# * c**(self.freqs+1+self.c_power_offset)
        # ic((self.freqs+1+self.c_power_offset).min())

        coeffs = (self.theta_coeffs.unsqueeze(0) * pow_matrix).sum(dim=(-1,-2))
        # ic(coeffs.dtype, mul.dtype, sinh_norm.dtype, powc.dtype, powkappa.dtype, powab.dtype, pow_matrix.dtype)
        coeffs = coeffs * mul
        # ic(pow_matrix.mean(), mul.mean(), sinh_norm.mean(), coeffs.mean(), powab.mean(), powc.mean(), powkappa.mean())
        return coeffs

class PhiISE(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        # pad coefficients to cube matrix
        self.max_degree = max_degree
        degrees = torch.arange(0, max_degree+1)
        freqs = 2**(degrees)
        self.max_sqkappa_pow = PHI_COEFFS.shape[1]
        assert(PHI_COEFFS.shape[1] == PHI_COEFFS.shape[2])
        diag_powers = torch.arange(self.max_sqkappa_pow**2).view(self.max_sqkappa_pow, self.max_sqkappa_pow) % (self.max_sqkappa_pow-1)
        kappa_powers = 2*diag_powers
        # coefficient dimensions go: (degree, polynomial in kappa^2, polynomial in c^2)
        phi_coeffs = PHI_COEFFS[:self.max_degree+1]
        # rearrange coefficients. 0,0 corresponds to k^0. (0, 1), (1, 0) correspond to k^2 * (c^0, c^2) respectively
        # row corresponds to power in c^0. column corresponds to power in (a^2+b^2)
        # each diagonal entry is a tensor of coefficients for a polynomial in kappa^2
        re_phi_coeffs = torch.zeros(self.max_degree+1, self.max_sqkappa_pow, self.max_sqkappa_pow)
        for deg in range(self.max_degree+1):
            for kap_pow in range(self.max_sqkappa_pow):
                for c_pow in range(self.max_sqkappa_pow):
                    coeff = phi_coeffs[deg, kap_pow, c_pow]
                    a2b2pow = kap_pow - c_pow
                    re_phi_coeffs[deg, c_pow, a2b2pow] = coeff

        
        mul = torch.tensor([math.log(math.factorial(f)) for f in freqs]).reshape(1, -1)
        abc_pow = torch.arange(self.max_sqkappa_pow)
        self.register_buffer('phi_coeffs', re_phi_coeffs)
        self.register_buffer('kappa_powers', kappa_powers)
        self.register_buffer('freqs', freqs)
        self.register_buffer('mul', mul)
        self.register_buffer('abc_pow', abc_pow)
        
        #format: k is the degree
        # prefix = (a-ib)^k kappa^k  / sinh(kappa)
        # kappa^(2*j) * self.coeffs[k, j] * (a^2+b^2)^j c^(k-j)
        
    def forward(self, vec, kappa):
        B = vec.shape[0]
        kappa = kappa.reshape(B, 1)
        a, b, c = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
        lpowkappa = kappa.log().reshape(B, 1, 1)*self.kappa_powers.unsqueeze(0) - (torch.sinh(kappa.reshape(B, 1, 1))+1e-8).log()
        powkappa = lpowkappa.exp()
        # powkappa = kappa.reshape(B, 1, 1)**self.kappa_powers.unsqueeze(0)
        powc = c**(2*self.abc_pow)
        powab = (a**2+b**2)**self.abc_pow

        # we have this triangle of powers in c^2 and (a^2+b^2) and kappa^2
        # 1
        # k^2 [(a^2+b^2)^1 c^0 + c^2 (a^2+b^2)^0]
        # k^4 [(a^2+b^2)^2 c^0 + c^2 (a^2+b^2)^1 + c^4 (a^2+b^2)^0]
        # This is then padded out to the following for the coefficients
        #   1 [              0,                 0,               0]
        # k^2 [(a^2+b^2)^1 c^0,   c^2 (a^2+b^2)^0,               0]
        # k^4 [(a^2+b^2)^2 c^0,   c^2 (a^2+b^2)^1, c^4 (a^2+b^2)^0]
        # the first dimension is the degree, but these things are duplicated across the degree
        pow_matrix = powkappa * powab.reshape(B, 1, -1) * powc.reshape(B, -1, 1)

        mul = (kappa.log()*(self.freqs+1) - self.mul).exp() * (a-torch.tensor(1j, dtype=torch.cfloat, device=vec.device)*b)**(self.freqs)

        coeffs = (self.phi_coeffs.unsqueeze(0) * pow_matrix.unsqueeze(1)).sum(dim=(-1,-2)) * mul
        return coeffs

class ISE(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        self.phi_ise = PhiISE(max_degree)
        self.theta_ise = ThetaISE(max_degree)
        
    def forward(self, vec, kappa):
        v = 200
        kappa = kappa.clip(0, 40)
        horz_coeffs = self.phi_ise(vec, kappa)
        vert_coeffs = self.theta_ise(vec, kappa)
        # ic(horz_coeffs.abs().max(), vert_coeffs.abs().max(), kappa.min(), kappa.max())
        return torch.stack([
            vert_coeffs.real.clip(min=-v,max=v),
            vert_coeffs.imag.clip(min=-v,max=v),
            horz_coeffs.real.clip(min=-v,max=v),
            horz_coeffs.imag.clip(min=-v,max=v),
        ], dim=1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    res = 100
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
    
    max_deg = 4
    ise = ISE(max_deg)
    # coeffs = ise(ang_vecs, 1*torch.ones(ang_vecs.shape[0]))
    coeffs = ise(ang_vecs, 20*torch.ones(ang_vecs.shape[0]))
    ic(coeffs.shape)

    for deg in range(max_deg):
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(coeffs[:, 0, deg].reshape(res, 2*res))
        ax[0, 1].imshow(coeffs[:, 1, deg].reshape(res, 2*res))
        ax[1, 0].imshow(coeffs[:, 2, deg].reshape(res, 2*res))
        ax[1, 1].imshow(coeffs[:, 3, deg].reshape(res, 2*res))

    plt.show()