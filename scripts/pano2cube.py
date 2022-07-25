from models import render_modules, tonemap, ish, bg_modules
import imageio
from icecream import ic
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

batch_size = 4096*500
device = torch.device('cuda')
epochs = 100

# bg_module = render_modules.BackgroundRender(3, render_modules.PanoUnwrap(), bg_resolution=2*1024, featureC=128, num_layers=0)
# bg_module = render_modules.BackgroundRender(3, render_modules.CubeUnwrap(), bg_resolution=2*1024, featureC=128, num_layers=0)
tm = tonemap.SRGBTonemap()
# bg_module = render_modules.HierarchicalBG(3, render_modules.CubeUnwrap(), bg_resolution=2*1024//4**5, num_levels=6, featureC=128, num_layers=0)
bg_module = bg_modules.HierarchicalBG(3, bg_modules.CubeUnwrap(), bg_resolution=2*1024//2**4, num_levels=5, featureC=128, num_layers=0, activation='softplus', power=2)
# bg_module = render_modules.MLPRender_FP(0, None, ish.ListISH([0,1,2,4,8,16]), -1, 256, 6)
ic(bg_module)
bg_module = bg_module.to(device)
pano = imageio.imread("ninomaru_teien_4k.exr")
optim = torch.optim.Adam(bg_module.parameters(), lr=0.80)
# optim = torch.optim.Adam(bg_module.parameters(), lr=1.0)
# optim = torch.optim.SGD(bg_module.parameters(), lr=0.5, momentum=0.99, weight_decay=0)
# optim = torch.optim.Adam(bg_module.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.94)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
# ic(bg_module.bg_mats[-1].shape, pano.shape)

H, W, C = pano.shape
rows, cols = torch.meshgrid(
    torch.arange(H, device=device),
    torch.arange(W, device=device),
    indexing='ij')

rows = rows.reshape(-1)
cols = cols.reshape(-1)

colors = torch.tensor(pano, dtype=torch.float32, device=device)
# colors = tm(colors, noclip=True)
# ic(bg_module.bg_mat.shape, colors.shape)

# bg_module.bg_mat = torch.nn.Parameter(torch.flip(colors, dims=[0]).permute(2, 0, 1).reshape(1, C, H, W))
# bg_module.bg_mat = torch.nn.Parameter(colors.permute(2, 0, 1).reshape(1, C, H, W))
col_mat = colors.permute(2, 0, 1).unsqueeze(0)
colors = colors.reshape(-1, 3)
N = colors.shape[0]
# ic(bg_module.bg_mats[-1].numel(), N)

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self, batch=None):
        batch = self.batch if batch is None else batch
        self.curr+=batch
        if self.curr + batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        ids = self.ids[self.curr:self.curr+batch]
        return ids

kappa = torch.tensor(20, device=device)
sampler = SimpleSampler(N, batch_size)
iter = tqdm(range(epochs))
for i in iter:
    inds = sampler.nextids()
    samp = colors[inds]
    # theta = (rows[inds]+0.5+0*torch.rand(batch_size, device=device))/H * np.pi - np.pi/2
    # phi = -(cols[inds]+0.5+0*torch.rand(batch_size, device=device))/W * 2*np.pi - np.pi
    brows = rows[inds]#+torch.rand(batch_size, device=device)
    bcols = cols[inds]#+torch.rand(batch_size, device=device)
    theta = brows/(H-1) * np.pi - np.pi/2
    phi = -bcols/(W-1) * 2*np.pi - np.pi

    # TODO REMOVE
    # theta = (rows[inds]+0.5+0*torch.rand(batch_size, device=device))/H * np.pi - np.pi/2
    # phi = (cols[inds]+0.5+0*torch.rand(batch_size, device=device))/W * 2*np.pi
    # ic(rows[inds], cols[inds], theta)

    vecs = torch.stack([
        torch.cos(phi)*torch.cos(theta),
        torch.sin(phi)*torch.cos(theta),
        -torch.sin(theta),
    ], dim=1)
    samp_vecs = vecs
    # roughness = 1e-8*torch.ones(theta.shape[0], device=device)
    roughness = torch.ones(theta.shape[0], device=device)*0.1**(10*i/epochs+3)
    # roughness = 99999*torch.ones(theta.shape[0], device=device)
    output = bg_module(samp_vecs, roughness)
    # viewdotnorm = torch.ones_like(theta).reshape(-1, 1)
    # roughness = 0.01*torch.ones_like(theta).reshape(-1, 1)
    # output = bg_module(pts=torch.zeros_like(vecs), viewdirs=None, features=None, refdirs=samp_vecs, roughness=roughness, viewdotnorm=viewdotnorm)

    loss = torch.sqrt((output - samp)**2+1e-8).mean()
    photo_loss = torch.sqrt((output.clip(0, 1) - samp.clip(0, 1))**2+1e-8).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
    psnr = -10.0 * np.log(photo_loss.detach().item()) / np.log(10.0)
    iter.set_description(f"PSNR: {psnr}. LR: {scheduler.get_last_lr()}")
    scheduler.step()

# bg_resolution = bg_module.bg_mats[-1].shape[2] // bg_module.unwrap_fn.H_mul
# bg_mat = 0
# for i, mat in enumerate(bg_module.bg_mats):
#     bg_mat2 = F.interpolate(mat.data, size=(bg_resolution*bg_module.unwrap_fn.H_mul, bg_resolution*bg_module.unwrap_fn.W_mul), mode='bilinear', align_corners=bg_module.align_corners).cpu()
#     # bg_mat += bg_mat2 / 2**i
#     bg_mat += bg_mat2 / (i+1)
#     plt.imshow(F.softplus(bg_mat[0].permute(1, 2, 0)))
#     plt.show()
bg_module.save('log/cubed.png')
torch.save(bg_module.state_dict(), 'log/mats360_bg.th')
# torch.save(bg_module.state_dict(), 'log/refmodule_mats360.th')

