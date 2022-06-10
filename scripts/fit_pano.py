from operator import index
from models.ish import ISH, RandISH, FullISH
from models.ise import RandISE
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
from models.sh import eval_sh_bases
import tinycudann as tcnn

pano_path = "/data/sun_pano/streetview/2 1.jpg"
device = torch.device('cuda')
pano = imageio.imread(pano_path)
H, W, C = pano.shape
pano_t = torch.tensor(pano, dtype=torch.float32, device=device)/255
rows, cols = torch.meshgrid(
        torch.arange(pano_t.shape[0]),
        torch.arange(pano_t.shape[1]),
        indexing='ij')
rows = rows.reshape(-1)
cols = cols.reshape(-1)
colors = pano_t.reshape(-1, 3)
N = colors.shape[0]

theta = rows/H * np.pi - np.pi/2
phi = cols/W * 2*np.pi - np.pi

angs = torch.stack([theta, phi], dim=1).reshape(-1, 2).to(device)

vecs = torch.stack([
    torch.cos(phi)*torch.cos(theta),
    torch.sin(phi)*torch.cos(theta),
    torch.sin(theta),
], dim=1).to(device)

# ish = ISH(4)
ish = RandISH(1024, 4).to(device)
# ish = RandISE(512, 32).to(device)
# ish = FullISH(4).to(device)
featureC = 128

def init_weights(m):
    if isinstance(m, torch.nn.Linear) and m.weight.shape[1] > 200:
        torch.nn.init.xavier_uniform_(m.weight, gain=0.2688)

mlp = torch.nn.Sequential(
    torch.nn.Linear(ish.dim(), featureC),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(featureC, featureC),
    torch.nn.ReLU(inplace=True),
    # torch.nn.Linear(featureC, featureC),
    # torch.nn.ReLU(inplace=True),
    # torch.nn.Linear(featureC, featureC),
    # torch.nn.ReLU(inplace=True),
    torch.nn.Linear(featureC, featureC),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(featureC, 3),
    torch.nn.Sigmoid()
).to(device)

mlp = torch.nn.Sequential(
    torch.nn.Linear(ish.dim()+25, 3),
    torch.nn.Sigmoid()
).to(device)

encoding = tcnn.Encoding(3, dict(
    otype="HashGrid",
    n_levels=16,
    n_features_per_level=2,
    log2_hashmap_size=14,
    base_resolution=1,
    per_level_scale=2
))
network = tcnn.Network(encoding.n_output_dims, 3, dict(
    otype="FullyFusedMLP",
    activation="ReLU",
    output_activation="None",
    n_neurons=64,
    n_hidden_layers=2,
))
mlp = torch.nn.Sequential(
    encoding, network
).to(device)

mlp.apply(init_weights)

optim = torch.optim.Adam(mlp.parameters(), lr=3e-3)
batch_size = 4096*100

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

kappa = torch.tensor(10, device=device)
sampler = SimpleSampler(N, batch_size)
for i in tqdm(range(500)):
    inds = sampler.nextids()
    samp = colors[inds]
    samp_vecs = vecs[inds]
    samp_angs = angs[inds]

    # enc = ish(samp_vecs, kappa).reshape(batch_size, -1)
    # enc = torch.cat([enc, eval_sh_bases(4, samp_vecs)], dim=-1)
    # output = mlp(enc)

    output = mlp(samp_vecs)

    loss = ((output - samp.half())**2).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()

# kappa = torch.tensor(20, device=device)
outs = []
with torch.no_grad():
    for i in tqdm(range(0, N, batch_size)):
        torch.cuda.empty_cache()
        inds = torch.arange(i, min(N, i+batch_size), device=device)
        samp_vecs = vecs[inds]
        samp_angs = angs[inds]

        # enc = ish(samp_vecs, kappa).reshape(-1, ish.dim())
        # enc = torch.cat([enc, eval_sh_bases(4, samp_vecs)], dim=-1)
        # output = mlp(enc)

        output = mlp(samp_vecs)

        outs.append(output.cpu())
        del output
output = torch.cat(outs, dim=0).reshape(H, W, 3)
plt.imshow(output.float())
plt.show()