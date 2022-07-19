from models import render_modules, tonemap
import imageio
from icecream import ic
import torch
import numpy as np
from tqdm import tqdm

batch_size = 4096*500
device = torch.device('cuda')
epochs = 500

# bg_module = render_modules.BackgroundRender(3, render_modules.PanoUnwrap(), bg_resolution=2*1024, featureC=128, num_layers=0)
# bg_module = render_modules.BackgroundRender(3, render_modules.CubeUnwrap(), bg_resolution=2*1024, featureC=128, num_layers=0)
tm = tonemap.SRGBTonemap()
bg_module = render_modules.HierarchicalBG(3, render_modules.CubeUnwrap(), bg_resolution=2*1024//2, num_levels=3, featureC=128, num_layers=0)
bg_module = bg_module.to(device)
pano = imageio.imread("ninomaru_teien_4k.exr")
# optim = torch.optim.Adam(bg_module.parameters(), lr=0.30)
optim = torch.optim.Adam(bg_module.parameters(), lr=0.1)
# optim = torch.optim.SGD(bg_module.parameters(), lr=1.0, momentum=0.99, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.93)
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
    theta = (rows[inds]+0.5+0*torch.rand(batch_size, device=device))/H * np.pi - np.pi/2
    phi = -(cols[inds]+0.5+0*torch.rand(batch_size, device=device))/W * 2*np.pi - np.pi
    # theta = (rows[inds]+torch.rand(batch_size, device=device))/H * np.pi - np.pi/2
    # phi = -(cols[inds]+torch.rand(batch_size, device=device))/W * 2*np.pi - np.pi
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
    output = bg_module(samp_vecs)
    loss = torch.sqrt((output - samp)**2+1e-8).mean()
    photo_loss = torch.sqrt((output.clip(0, 1) - samp.clip(0, 1))**2+1e-8).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
    psnr = -10.0 * np.log(photo_loss.detach().item()) / np.log(10.0)
    iter.set_description(f"PSNR: {psnr}. LR: {scheduler.get_last_lr()}")
    scheduler.step()

bg_module.save('cubed.png')
torch.save(bg_module.state_dict(), 'log/mats360_bg.th')

