import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from models import safemath
from icecream import ic

res = 80
device = 'cpu'
degree = 16
M = 1
N = 2*res**2

def von_mises_fisher(cos_dist, kappa):
    return torch.exp(cos_dist * kappa) / (4 * np.pi * np.sinh(kappa))

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
], dim=-1).to(device).reshape(-1, 3)

# N x N
cosdists = ang_vecs @ ang_vecs.T

kappas = torch.linspace(1/np.sqrt(0.1/180*np.pi), 1/np.sqrt(30/180*np.pi), M, dtype=torch.float32).to(device)

def spherical_encoding(refdirs, roughness, pe, ind_order=[0, 1, 2]):
    i, j, k = ind_order
    return [safemath.integrated_pos_enc((refangs[..., 0:1], roughness), 0, pe),
            safemath.integrated_pos_enc((refangs[..., 1:2], roughness), 0, pe)]

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vec_layer = torch.nn.Linear(6, 6)
        self.layer2 = torch.nn.Linear(6, 1)
        self.kappa_layer1 = torch.nn.Linear(1, 1)
        self.kappa_layer2 = torch.nn.Linear(1, 1)
        self.register_parameter('exponent1', torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('exponent2', torch.nn.Parameter(torch.tensor(1.0)))
        
    def forward(self, x):
        kappa = x[:, 0][:, None]
        vecs = x[:, 1:]

        i, j, k = 0, 1, 2
        norm2d = torch.sqrt(vecs[..., i]**2+vecs[..., j]**2)
        refangs = torch.stack([
            safemath.atan2(vecs[..., j], vecs[..., i]),
            safemath.atan2(vecs[..., k], norm2d),
        ], dim=-1)
        x = torch.cat([x, refangs], dim=-1)

        k2 = self.kappa_layer2(kappa)#**self.exponent2
        k1 = self.kappa_layer1(kappa)#**self.exponent1
        return self.layer2(torch.cos(self.vec_layer(x))) * kappa / (2*np.pi*torch.sinh(kappa))

for i, deg in enumerate(range(1, degree)):
    exp_cos_azis = []
    exp_cos_eles = []
    exp_sin_azis = []
    exp_sin_eles = []
    for kappa in tqdm(kappas):
        probs = von_mises_fisher(cosdists, kappa)
        # probs = probs / probs.sum(dim=1, keepdim=True)

        # (N)
        expazi = (torch.exp(-1j * 2**deg * azis).reshape(1, N)*probs).sum(dim=-1)
        expele = (torch.exp(-1j * 2**deg * eles).reshape(1, N)*probs).sum(dim=-1)
        exp_cos_azi = expazi.real
        exp_sin_azi = expazi.imag
        exp_cos_ele = expele.real
        exp_sin_ele = expele.imag
        
        exp_cos_azis.append(exp_cos_azi)
        exp_cos_eles.append(exp_cos_ele)
        exp_sin_azis.append(exp_sin_azi)
        exp_sin_eles.append(exp_sin_ele)
        ic(kappa, deg)
        # plt.imshow(exp_cos_azi.reshape(res, 2*res))
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(exp_cos_azi.reshape(res, 2*res))
        ax[0, 1].imshow(exp_cos_ele.reshape(res, 2*res))
        ax[1, 0].imshow(exp_sin_azi.reshape(res, 2*res))
        ax[1, 1].imshow(exp_sin_ele.reshape(res, 2*res))
        plt.show()
        
    exp_cos_azis = torch.stack(exp_cos_azis, dim=0)
    exp_cos_azis_f = exp_cos_azis.reshape(-1, 1).cuda()

    """
    inputs = torch.cat([
        kappas.reshape(-1, 1, 1).expand(M, N, 1),
        ang_vecs.reshape(1, -1, 3).expand(M, N, 3),
    ], dim=2).reshape(-1, 4).cuda()
    m = Model()
    m.cuda()
    optim = torch.optim.Adam(m.parameters(), lr=1e-2)
    lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, factor=0.90)
    for i in range(3000):
        out = m(inputs)
        loss = ((out - exp_cos_azis_f)**2).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_sch.step(loss)
        lr = optim.param_groups[0]['lr']
        if lr < 1e-8:
            break
        if i % 100 == 0:
            print(loss.item())
        
    # plt.scatter(exp_cos_azis, m(inputs).detach() - exp_cos_azis)
    # plt.scatter(exp_cos_azis.cpu(), m(inputs).detach().cpu())
    # plt.show()
    pred_outs = m(inputs).detach().cpu().reshape(M, res, 2*res)
    """
    # fig, ax = plt.subplots(2, M)
    # for i in range(M):
    #     ax[0, i].imshow(pred_outs[i])
    #     # ax[1, i].imshow(exp_cos_azis[i].reshape(res, 2*res))
    #     ax[1, i].imshow(exp_cos_eles[i].reshape(res, 2*res))
    # plt.show()
        
    
    