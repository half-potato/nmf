from operator import index
from models.ish import ISH, RandISH, FullISH, ListISH, RandRotISH
from models.ise import RandISE
import cv2
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
from models.sh import eval_sh_bases
#  import tinycudann as tcnn
from pathlib import Path

device = torch.device('cuda')
#  pano_paths = [
#      "/data/sun_pano/streetview/2 1.jpg",
#      "/data/sun_pano/streetview/15 1.jpg",
#      "/data/sun_pano/streetview/16 1.jpg",
#      "/data/sun_pano/streetview/17 1.jpg",
#      "/data/sun_pano/streetview/18 1.jpg",
#  ]
pano_paths = list(Path('./envmaps/').glob('*.png'))
pano_paths = [Path('./ninomaru_teien_4k.exr')]
hdr = True
ish = ListISH([1,2,4,8,16]).to(device)
# ish = RandRotISH(16, [1,2,8], [16]).to(device)
# ish = RandISH(256, 10).to(device)
ic(ish.dim())
batch_size = 4096*50

# ish = RandISE(512, 32).to(device)
# ish = FullISH(4).to(device)
featureC = 256
#  featureC = 128
#  pano_feat_dim = 128
pano_feat_dim = 27*3
#  num_layers = 3
num_layers = 6
epochs = 500

num_panos = len(pano_paths)
colors = []
for pano_path in pano_paths:
    pano = imageio.imread(pano_path)
    H, W, C = pano.shape
    pano = cv2.resize(pano, (2000, 1000))
    # pano = cv2.resize(pano, (200, 100))
    H, W, C = pano.shape
    pano_t = torch.tensor(pano, dtype=torch.float32, device=device)
    if not hdr:
        pano_t = pano_t/255
    else:
        pano_t = pano_t.clip(0, 1)
    colors.append(pano_t.reshape(-1, 3))

M = colors[0].shape[0]
colors = torch.cat(colors, dim=0)
N = colors.shape[0]

rows, cols = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing='ij')
rows = rows.reshape(-1)
cols = cols.reshape(-1)
theta = rows/H * np.pi - np.pi/2
phi = -cols/W * 2*np.pi - np.pi

angs = torch.stack([theta, phi], dim=1).reshape(-1, 2).to(device)

vecs = torch.stack([
    torch.cos(phi)*torch.cos(theta),
    torch.sin(phi)*torch.cos(theta),
    -torch.sin(theta),
], dim=1).to(device)

# ish = ISH(4)

pano_features = torch.rand(num_panos, pano_feat_dim, device=device)
pano_features.requires_grad=True

def get_features(inds):
    pano_ind = inds // M
    return pano_features[pano_ind]

def init_weights(m):
    if isinstance(m, torch.nn.Linear):# and m.weight.shape[1] > 200:
        torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

mlp = torch.nn.Sequential(
    torch.nn.Linear(ish.dim()+pano_feat_dim, featureC),
    *sum([[
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC, bias=False)
        ] for _ in range(num_layers-2)], []),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(featureC, 3),
    torch.nn.Sigmoid()
).to(device)

#  mlp = torch.nn.Sequential(
#      torch.nn.Linear(ish.dim()+25, 3),
#      torch.nn.Sigmoid()
#  ).to(device)

"""
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
"""

mlp.apply(init_weights)

optim = torch.optim.Adam(mlp.parameters(), lr=5e-4)

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

sampler = SimpleSampler(N, batch_size)
for i in tqdm(range(epochs)):
    inds = sampler.nextids()
    samp = colors[inds]
    samp_vecs = vecs[inds % M]
    samp_angs = angs[inds % M]
    batch_size = samp_angs.shape[0]
    roughness = 0.01*torch.ones((batch_size, 1), device=device)

    enc = ish(samp_vecs, roughness)
    enc = enc.reshape(batch_size, -1)
    feats = get_features(inds)
    enc = torch.cat([enc, feats], dim=-1)
    #  enc = torch.cat([enc, eval_sh_bases(4, samp_vecs)], dim=-1)
    output = mlp(enc)

    #  output = mlp(samp_vecs)
    #  loss = ((output - samp.half())**2).mean()

    loss = ((output - samp)**2).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
ic(loss)

# kappa = torch.tensor(20, device=device)
with torch.no_grad():
    for pind in range(num_panos):
        outs = []
        for i in tqdm(range(0, M, batch_size)):
            torch.cuda.empty_cache()
            inds = torch.arange(i, min(M, i+batch_size), device=device) + pind * M
            roughness = 0.01*torch.ones((inds.shape[0], 1), device=device)
            samp_vecs = vecs[inds % M]
            samp_angs = angs[inds % M]

            enc = ish(samp_vecs, roughness).reshape(-1, ish.dim())
            #  enc = torch.cat([enc, eval_sh_bases(4, samp_vecs)], dim=-1)
            feats = get_features(inds)
            enc = torch.cat([enc, feats], dim=-1)
            output = mlp(enc)

            #  output = mlp(samp_vecs)

            outs.append(output.cpu())
            del output
        output = torch.cat(outs, dim=0)
        pano = output.reshape(H, W, 3)


        pano = (pano*255).numpy().astype(np.uint8)
        pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'fit_panos/pano{pind}.png', pano)
