from pathlib import Path
import re
import numpy as np
import torch
from icecream import ic
import math
from modules import safemath

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
   {{{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0.}, {0. - 0.0981748 I, 0., 0. + 0.0981748 I, 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0.}}, {{0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0.}, {0. - 0.133333 I, 0., 
   0. + 0.133333 I, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0.}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0.}}, {{0., -0.0245437, 0., 0.0245437, 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0.}, {0. - 0.0031746 I, 0., 0. + 0.00846561 I, 
   0., 0. - 0.00529101 I, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0., -0.00076699, 0., 0.000511327, 0., 0.000255663, 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0.}}, {{0., -2.66316*10^-6, 0., 
   0.000015979, 0., -0.0000282295, 0., 0.0000149137, 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0.}, {0. - 1.26961*10^-7 I, 0., 0. + 1.45098*10^-6 I, 
   0., 0. - 4.17883*10^-6 I, 0., 0. + 3.71451*10^-6 I, 0., 
   0. - 8.59707*10^-7 I, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0., -6.24178*10^-8, 0., 3.60636*10^-7, 0., -5.74244*10^-7, 
   0., 2.02115*10^-7, 0., 7.39106*10^-8, 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}}, {{0., -5.75916*10^-17, 0., 1.3438*10^-15, 
   0., -1.13686*10^-14, 0., 4.67644*10^-14, 0., -1.03966*10^-13, 0., 
   1.27612*10^-13, 0., -8.13398*10^-14, 0., 2.1011*10^-14, 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0. - 9.92344*10^-19 I, 0., 0. + 4.4457*10^-17 I, 0., 
   0. - 5.74522*10^-16 I, 0., 0. + 3.28298*10^-15 I, 0., 
   0. - 9.72735*10^-15 I, 0., 0. + 1.57659*10^-14 I, 0., 
   0. - 1.35829*10^-14 I, 0., 0. + 5.17444*10^-15 I, 0., 
   0. - 3.81988*10^-16 I, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0.}, {0., -7.87385*10^-19, 0., 1.82973*10^-17, 
   0., -1.53068*10^-16, 0., 6.18569*10^-16, 0., -1.33733*10^-15, 0., 
   1.56351*10^-15, 0., -8.96945*10^-16, 0., 1.54456*10^-16, 0., 
   3.32988*10^-17, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0.}}, {{0., -1.99187*10^-43, 0., 1.79268*10^-41, 
   0., -6.20467*10^-40, 0., 1.12381*10^-38, 0., -1.22876*10^-37, 0., 
   8.80622*10^-37, 0., -4.35474*10^-36, 0., 1.53505*10^-35, 
   0., -3.93293*10^-35, 0., 7.38977*10^-35, 0., -1.01729*10^-34, 0., 
   1.01361*10^-34, 0., -7.11084*10^-35, 0., 3.32968*10^-35, 
   0., -9.33841*10^-36, 0., 1.18613*10^-36, 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}, {0. - 1.22731*10^-45 I, 0., 0. + 2.15373*10^-43 I, 0., 
   0. - 1.12291*10^-41 I, 0., 0. + 2.73935*10^-40 I, 0., 
   0. - 3.79856*10^-39 I, 0., 0. + 3.32952*10^-38 I, 0., 
   0. - 1.96763*10^-37 I, 0., 0. + 8.1664*10^-37 I, 0., 
   0. - 2.44144*10^-36 I, 0., 0. + 5.33024*10^-36 I, 0., 
   0. - 8.52839*10^-36 I, 0., 0. + 9.92885*10^-36 I, 0., 
   0. - 8.23727*10^-36 I, 0., 0. + 4.66678*10^-36 I, 0., 
   0. - 1.65521*10^-36 I, 0., 0. + 2.9426*10^-37 I, 0., 
   0. - 7.45605*10^-39 I, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0.}, {0., -1.45889*10^-45, 0., 1.31235*10^-43, 
   0., -4.53228*10^-42, 0., 8.18612*10^-41, 0., -8.92089*10^-40, 0., 
   6.36812*10^-39, 0., -3.13409*10^-38, 0., 1.09831*10^-37, 
   0., -2.79327*10^-37, 0., 5.19843*10^-37, 0., -7.06453*10^-37, 0., 
   6.91095*10^-37, 0., -4.71362*10^-37, 0., 2.10207*10^-37, 
   0., -5.30591*10^-38, 0., 4.49291*10^-39, 0., 5.20569*10^-40, 0., 
   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
   0.}}, {{0., -2.73084*10^-106, 0., 9.59437*10^-104, 
   0., -1.32487*10^-101, 0., 9.74296*10^-100, 0., -4.42549*10^-98, 0.,
    1.35782*10^-96, 0., -2.98721*10^-95, 0., 4.91644*10^-94, 
   0., -6.24677*10^-93, 0., 6.27636*10^-92, 0., -5.08087*10^-91, 0., 
   3.36331*10^-90, 0., -1.84206*10^-89, 0., 8.42572*10^-89, 
   0., -3.24245*10^-88, 0., 1.05576*10^-87, 0., -2.92074*10^-87, 0., 
   6.8846*10^-87, 0., -1.38475*10^-86, 0., 2.37716*10^-86, 
   0., -3.47877*10^-86, 0., 4.32824*10^-86, 0., -4.55825*10^-86, 0., 
   4.03696*10^-86, 0., -2.97911*10^-86, 0., 1.80858*10^-86, 
   0., -8.8723*10^-87, 0., 3.42793*10^-87, 0., -1.0037*10^-87, 0., 
   2.09246*10^-88, 0., -2.76635*10^-89, 0., 1.74269*10^-90, 0., 
   0.}, {0. - 5.98343*10^-109 I, 0., 0. + 4.14547*10^-106 I, 0., 
   0. - 8.5954*10^-104 I, 0., 0. + 8.44972*10^-102 I, 0., 
   0. - 4.81486*10^-100 I, 0., 0. + 1.78091*10^-98 I, 0., 
   0. - 4.59676*10^-97 I, 0., 0. + 8.70423*10^-96 I, 0., 
   0. - 1.25391*10^-94 I, 0., 0. + 1.41239*10^-93 I, 0., 
   0. - 1.27053*10^-92 I, 0., 0. + 9.28036*10^-92 I, 0., 
   0. - 5.57727*10^-91 I, 0., 0. + 2.7868*10^-90 I, 0., 
   0. - 1.16741*10^-89 I, 0., 0. + 4.12649*10^-89 I, 0., 
   0. - 1.23681*10^-88 I, 0., 0. + 3.15422*10^-88 I, 0., 
   0. - 6.85912*10^-88 I, 0., 0. + 1.27288*10^-87 I, 0., 
   0. - 2.01475*10^-87 I, 0., 0. + 2.71467*10^-87 I, 0., 
   0. - 3.10248*10^-87 I, 0., 0. + 2.99085*10^-87 I, 0., 
   0. - 2.41278*10^-87 I, 0., 0. + 1.61079*10^-87 I, 0., 
   0. - 8.76163*10^-88 I, 0., 0. + 3.79751*10^-88 I, 0., 
   0. - 1.26901*10^-88 I, 0., 0. + 3.10233*10^-89 I, 0., 
   0. - 5.0451*10^-90 I, 0., 0. + 4.33987*10^-91 I, 0., 
   0. - 3.85217*10^-93 I, 0.}, {0., -1.0334*10^-108, 0., 
   3.63046*10^-106, 0., -5.0107*10^-104, 0., 3.68268*10^-102, 
   0., -1.67172*10^-100, 0., 5.12568*10^-99, 0., -1.12683*10^-97, 0., 
   1.85312*10^-96, 0., -2.35255*10^-95, 0., 2.36151*10^-94, 
   0., -1.90977*10^-93, 0., 1.26278*10^-92, 0., -6.90773*10^-92, 0., 
   3.15538*10^-91, 0., -1.21245*10^-90, 0., 3.94118*10^-90, 
   0., -1.08826*10^-89, 0., 2.55966*10^-89, 0., -5.13577*10^-89, 0., 
   8.79128*10^-89, 0., -1.28224*10^-88, 0., 1.58904*10^-88, 
   0., -1.66551*10^-88, 0., 1.46642*10^-88, 0., -1.0742*10^-88, 0., 
   6.45923*10^-89, 0., -3.12806*10^-89, 0., 1.18659*10^-89, 
   0., -3.37785*10^-90, 0., 6.70662*10^-91, 0., -7.97985*10^-92, 0., 
   3.35132*10^-93, 0., 2.01548*10^-94}}}
""")

class ThetaISE(torch.nn.Module):
    def __init__(self, max_degree=1, maxv=200):
        super().__init__()
        self.maxv = maxv
        # pad coefficients to cube matrix
        self.max_degree = max_degree
        degrees = torch.arange(0, max_degree+1)
        freqs = 2**(degrees)
        # row = kappa degree, col = a^2 degree
        if max_degree > THETA_COEFFS.shape[0]:
            raise ValueError("max_degree must be less than {}".format(THETA_COEFFS.shape[0]))
        theta_coeffs = THETA_COEFFS[:self.max_degree+1]
        self.max_kappa_pow = theta_coeffs.shape[1]
        self.max_c_pow = theta_coeffs.shape[2]

        c_pow = torch.arange(self.max_c_pow)
        kappa_pow = torch.arange(self.max_kappa_pow)

        mul = torch.tensor([math.log(math.factorial(f)) for f in freqs]).reshape(1, -1)

        self.register_buffer('theta_coeffs', theta_coeffs.cfloat())
        self.register_buffer('mul', mul)
        self.register_buffer('c_pow', c_pow)
        self.register_buffer('kappa_pow', kappa_pow)
        self.register_buffer('freqs', freqs)

    def forward(self, vec, kappa):
        B = vec.shape[0]
        kappa = kappa.reshape(B, 1)
        a, b, c = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
        powkappa = kappa**self.kappa_pow
        powc = c.reshape(B, 1)**self.c_pow
        # ic(self.c_power_offset)
        # ic((self.c_powers-self.c_power_offset).min())
        pow_matrix = powkappa.reshape(B, -1, 1) * powc.reshape(B, 1, -1)

        sinh_norm = - (torch.sinh(kappa)+1e-8).log()
        # ic(kappa.log(), sinh_norm, (kappa.log()*(self.freqs+1) + sinh_norm))
        mul = (kappa.log()*(self.freqs+1) - self.mul + sinh_norm).exp()

        coeffs = (self.theta_coeffs.unsqueeze(0) * pow_matrix.unsqueeze(1)).sum(dim=(-1,-2))
        coeffs = coeffs * mul
        # ic(coeffs.mean())
        # ic(pow_matrix.mean(), mul.mean(), sinh_norm.mean(), coeffs.mean(), powc.mean(), powkappa.mean(), self.theta_coeffs.mean())
        return [
            coeffs.real,#.clip(min=-self.maxv,max=self.maxv),
            coeffs.imag,#.clip(min=-self.maxv,max=self.maxv),
        ]

class PhiISE(torch.nn.Module):
    def __init__(self, max_degree=1, maxv=200):
        super().__init__()
        self.maxv = maxv
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

        self.adjust = [0.8, 1.0]
        mul = torch.tensor([math.log(math.factorial(f))*self.adjust[0] for f in freqs]).reshape(1, -1)
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

        mul = (kappa.log()*(self.freqs*self.adjust[1]+1) - self.mul).exp() * (a-torch.tensor(1j, dtype=torch.cfloat, device=vec.device)*b)**(self.freqs)

        coeffs = (self.phi_coeffs.unsqueeze(0) * pow_matrix.unsqueeze(1)).sum(dim=(-1,-2)) * mul
        return [
            coeffs.real,#.clip(min=-self.maxv,max=self.maxv),
            coeffs.imag,#.clip(min=-self.maxv,max=self.maxv),
        ]

class PhiISEHack(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        # pad coefficients to cube matrix
        self.max_degree = max_degree
        scales = torch.tensor([2**i for i in range(0, self.max_degree)])
        self.register_buffer('scales', scales)

    def forward(self, vec, kappa):
        a, b, c = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
        phi = safemath.atan2(b, a)
        # theta = torch.acos(vec[..., k].clip(-1+1e-6, 1-1e-6)) - np.pi/2
        roughness = 1/(kappa**2+1e-8)
        # falloff = torch.exp(-c**16/(0.479426+0.4841333/(1+torch.exp(self.scales[None, :]-5)))**16)
        falloff = torch.exp(-self.scales[None, :] * c**4)
        # enc = safemath.integrated_pos_enc((x, roughness), 0, self.max_degree).reshape(-1, self.max_degree, 2)
        shape = list(phi.shape[:-1]) + [-1]
        y = torch.reshape(phi[..., None] * self.scales[None, :], shape)
        # ic(roughness.shape, scales.shape, y.shape)
        mul = torch.exp(-0.5 * roughness.reshape(-1, 1) * self.scales[None, :]**2) * falloff
        return [mul*torch.sin(y), mul*torch.cos(y)]

class ThetaISEHack(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        # pad coefficients to cube matrix
        self.max_degree = max_degree
        scales = torch.tensor([2**i for i in range(0, self.max_degree)])
        self.register_buffer('scales', scales)

    def forward(self, vec, kappa):
        a, b, c = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
        norm2d = torch.sqrt(a**2+b**2)
        radius = torch.sqrt(2-c**2)
        theta = safemath.atan2(c, norm2d)
        # theta = torch.acos(vec[..., k].clip(-1+1e-6, 1-1e-6)) - np.pi/2
        roughness = 1/(kappa**2+1e-8)
        falloff = c**2
        x = (theta*radius)[...]
        # enc = safemath.integrated_pos_enc((x, roughness), 0, self.max_degree).reshape(-1, self.max_degree, 2)
        shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x[..., None] * self.scales[None, :], shape)
        # ic(roughness.shape, scales.shape, y.shape)
        mul = torch.exp(-0.5 * roughness.reshape(-1, 1) * self.scales[None, :]**2) * falloff
        return [mul*torch.sin(y), mul*torch.cos(y)]


class RandISE(torch.nn.Module):
    def __init__(self, rand_n, std=10):
        super().__init__()
        r = 10
        theta_scales = torch.normal(0, std, (1, rand_n))
        phi_ratio = torch.normal(0, 2, (1, rand_n)).clip(min=-r, max=r)
        scales = torch.cat([
            theta_scales, phi_ratio*theta_scales
        ], dim=0)
        # scales[1].clip(scales[0])
        self.register_buffer('scales', scales)
        self.rand_n = rand_n

    def dim(self):
        return 4*self.rand_n

    def forward(self, vec, kappa):
        a, b, c = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
        norm2d = torch.sqrt(a**2+b**2)
        theta_radius = torch.sqrt(2-c**2)
        phi = safemath.atan2(b, a)
        theta = safemath.atan2(c, norm2d)
        # theta = torch.acos(vec[..., k].clip(-1+1e-6, 1-1e-6)) - np.pi/2
        roughness = 1/(kappa**2+1e-8)
        theta_falloff = c**2
        # enc = safemath.integrated_pos_enc((x, roughness), 0, self.max_degree).reshape(-1, self.max_degree, 2)
        x = torch.stack([
            phi, theta*theta_radius
        ], dim=-1)
        phi_falloff = torch.exp(-torch.abs(self.scales[None]) * c[:, :, None]**2)
        scaled_x = x @ self.scales
        # phi_falloff = torch.exp(-scaled_x[:, :]**4)
        sinx = torch.sin(scaled_x)
        cosx = torch.cos(scaled_x)
        scaled_roughness = torch.exp(-0.5 * roughness.reshape(-1, 1, 1) * self.scales[None]**2)
        # ic(scaled_roughness.mean(), phi_falloff.mean())
        return torch.cat([
            scaled_roughness * phi_falloff * sinx,
            scaled_roughness * phi_falloff * cosx,
        ], dim=-1)


class ISE(torch.nn.Module):
    def __init__(self, max_degree=1):
        super().__init__()
        # self.phi_ise = PhiISE(max_degree)
        self.phi_ise = PhiISEHack(max_degree)
        # self.theta_ise = PhiISE(max_degree)
        # self.theta_ise = ThetaISE(max_degree)
        self.theta_ise = ThetaISEHack(max_degree)
        # self.register_buffer('hc_mean', hc_mean)
        # self.register_buffer('hc_std', hc_std)
        
    def forward(self, vec, kappa):
        v = 200
        kappa = kappa.clip(1, 20)
        horz_coeffs = self.phi_ise(vec, kappa)
        vert_coeffs = self.theta_ise(vec, kappa)
        # ic(horz_coeffs.abs().max(), vert_coeffs.abs().max(), kappa.min(), kappa.max())
        # ic(horz_coeffs[0][..., 1:].mean(dim=0))
        # ic(horz_coeffs[1][..., 1:].mean(dim=0))
        # ic(horz_coeffs[0][..., 1:].std(dim=0))
        # ic(horz_coeffs[1][..., 1:].std(dim=0))
        # ic(self.hc_mean1.shape, horz_coeffs[0][..., 1:].shape, vert_coeffs[0].shape)
        # ic(kappa)
        # ic(((horz_coeffs[0][..., 1:]-self.hc_mean[2, None, :])/self.hc_std[2, None, :]),
        #     ((horz_coeffs[1][..., 1:]-self.hc_mean[3, None, :])/self.hc_std[3, None, :]))
        return torch.stack([
            vert_coeffs[0],
            vert_coeffs[1],
            # torch.sigmoid(vert_coeffs[0]/1),
            # torch.sigmoid(vert_coeffs[1]/1),
            # torch.sigmoid(horz_coeffs[0][..., 1:]),
            # torch.sigmoid(horz_coeffs[1][..., 1:]),
            # ((horz_coeffs[0][..., 1:]-self.hc_mean[2, None, :])/self.hc_std[2, None, :]),
            # ((horz_coeffs[1][..., 1:]-self.hc_mean[3, None, :])/self.hc_std[3, None, :]),
            # torch.sigmoid((horz_coeffs[0]-self.hc_mean[2, None, :])/self.hc_std[2, None, :]),
            # torch.sigmoid((horz_coeffs[1]-self.hc_mean[3, None, :])/self.hc_std[3, None, :]),
            horz_coeffs[0],
            horz_coeffs[1],
            # horz_coeffs[0][..., 1:],
            # horz_coeffs[1][..., 1:],
        ], dim=1)
        
    def dim(self):
        return (self.phi_ise.max_degree + self.theta_ise.max_degree)*2

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
    
    max_deg = 6
    # ise = ISE(max_deg)
    ise = RandISE(6)
    coeffs1 = ise(ang_vecs, 1*torch.ones(ang_vecs.shape[0]))
    coeffs = ise(ang_vecs, 15*torch.ones(ang_vecs.shape[0]))
    kcoeffs = torch.cat([coeffs1, coeffs], dim=0)
    # ic(torch.mean(kcoeffs, dim=0))
    # ic(torch.std(kcoeffs, dim=0))
    # ic(coeffs.shape)

    # for deg in range(max_deg):
    #     fig, ax = plt.subplots(2, 2)
    #     ic(coeffs.shape)
    #     ax[0, 0].imshow(coeffs[:, 0, deg].reshape(res, 2*res))
    #     ax[0, 1].imshow(coeffs[:, 1, deg].reshape(res, 2*res))
        # ax[1, 0].imshow(coeffs[:, 2, deg].reshape(res, 2*res))
        # ax[1, 1].imshow(coeffs[:, 3, deg].reshape(res, 2*res))

    for deg in range(coeffs.shape[1]):
        fig, ax = plt.subplots(coeffs.shape[2])
        for i in range(coeffs.shape[2]):
            ax[i].imshow(coeffs[:, deg, i].reshape(res, 2*res))

    plt.show()
