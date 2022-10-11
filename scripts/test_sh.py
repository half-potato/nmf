import torch
from models import bg_modules
import numpy as np
from models import sh
import matplotlib.pyplot as plt

def tm(x):
    return x / (x+1)

device = torch.device('cuda')
# bg_sd = torch.load('log/bg_mat2.pth')
bg_sd = torch.load('log/mats360_bg.th')
bg_module = bg_modules.HierarchicalCubeMap(bg_resolution=2048, num_levels=1, featureC=128, activation='softplus', power=2, lr=1e-2)
bg_module.load_state_dict(bg_sd, strict=False)
bg_module = bg_module.to(device)
mipval = -0

coeffs, conv_coeffs = bg_module.get_spherical_harmonics(50, mipval=mipval)
G = 100
_theta = torch.linspace(0, np.pi, G//2, device=device)
_phi = torch.linspace(0, 2*np.pi, G, device=device)
theta, phi = torch.meshgrid(_theta, _phi, indexing='ij')
sh_samples = torch.stack([
    torch.sin(theta) * torch.cos(phi),
    torch.sin(theta) * torch.sin(phi),
    torch.cos(theta),
], dim=-1)
fsamp = sh_samples.reshape(-1, 3)
evaled = sh.eval_sh_bases(coeffs.shape[0], fsamp)
# evaled = sh.sh_basis([0, 1, 2, 4, 8, 16], fsamp)
# cols = (conv_coeffs.reshape(1, -1, 3) * evaled.reshape(evaled.shape[0], -1, 1)).sum(dim=1)
cols = (coeffs.reshape(1, -1, 3) * evaled.reshape(evaled.shape[0], -1, 1)).sum(dim=1)
dcols = cols.reshape(*sh_samples.shape).detach().cpu()
plt.imshow(tm(dcols))

cols = (conv_coeffs.reshape(1, -1, 3) * evaled.reshape(evaled.shape[0], -1, 1)).sum(dim=1)
dcols = cols.reshape(*sh_samples.shape).detach().cpu()
plt.figure()
plt.imshow(tm(dcols))

cols = bg_module(fsamp, mipval*torch.ones_like(fsamp[:, 0]))
dcols = cols.reshape(*sh_samples.shape).detach().cpu()
plt.figure()
plt.imshow(tm(dcols))
plt.show()
