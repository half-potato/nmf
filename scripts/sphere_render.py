import os
from torch.nn.modules import activation
from tqdm.auto import tqdm
from models.tensor_nerf import TensorNeRF
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from omegaconf import DictConfig, OmegaConf
from models import render_modules

from dataLoader import dataset_dict
import sys
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import math
import numpy as np

renderer = chunk_renderer

class AddBasis(torch.nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[..., :self.n] + x[..., self.n:2*self.n] + x[..., 2*self.n:3*self.n]

@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent / 'configs'), config_name='default')
@torch.no_grad()
def main(cfg: DictConfig):
    device = torch.device('cuda')
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)
    print(cfg.dataset)
    print(cfg.model)
    print(cfg.params)

    if not os.path.exists(cfg.ckpt):
        print('the ckpt path does not exists!!')
        return

    # init model
    ckpt = torch.load(cfg.ckpt)
    ckpt['config']['bg_module']['bg_resolution'] = ckpt['state_dict']['bg_module.bg_mat'].shape[-1]
    tensorf = TensorNeRF.load(ckpt).to(device)
    tensorf.rf.set_smoothing(1)
    # tensorf.normal_module = render_modules.AppDimNormal(1, activation=torch.nn.Identity)
    tensorf.normal_module = render_modules.DeepMLPNormal(pospe=16, num_layers=3).to(device)
    tensorf.rf.basis_mat = AddBasis(48)
    tensorf.alphaMask = None

    # tensorf.rf.density_plane[i][:, :, 40:100, 40:100] += 1
    # tensorf.rf.density_line[i][:, :, ind:ind+8] += 100
    H, W = tensorf.rf.density_plane[0].shape[-2:]
    C = tensorf.rf.density_plane[0].shape[1]
    # tensorf.bg_module.bg_mat[..., :4000, :] = 0

    d = 1
    row, col, line = torch.meshgrid(torch.linspace(-d, d, H, device=device), torch.linspace(-d, d, W, device=device), torch.linspace(-d, d, H, device=device), indexing='ij')

    dist = torch.sqrt(row**2 + col**2 + line**2)

    optim = torch.optim.Adam(tensorf.parameters(), lr=1e-1)
    with torch.enable_grad():
        for _ in tqdm(range(50)):
            ox, oy, oz = torch.rand(3, H, W, H, device=device)/H
            y = (col+oy)
            x = (row+ox)
            z = (line+oz)
            xyz = torch.stack([x, y, z, torch.zeros_like(x)], dim=-1).reshape(-1, 4)
            dist = torch.sqrt(x**2 + y**2 + z**2).reshape(-1)
            mask = (dist < 0.2)
            feat = tensorf.rf.compute_densityfeature(xyz)
            sigma_feat = tensorf.feature2density(feat)

            world_loss = 0
            # """
            # shell = (dist < 0.2) & (dist > 0.19)
            # app_features = tensorf.rf.compute_appfeature(xyz[shell])
            # p_norm = tensorf.normal_module(xyz[shell], app_features)
            # gt_norm = xyz[shell, :3] / (torch.linalg.norm(xyz[shell, :3], dim=1, keepdim=True)+1e-10)
            # world_loss = -(p_norm * gt_norm).sum(dim=-1).sum()
            # """


            # sigma = 1-torch.exp(-sigma_feat * 0.025)
            sigma = 1-torch.exp(-sigma_feat)
            loss = (-sigma[mask].clip(max=200).sum() + sigma[~mask].clip(min=0).sum())
            loss = loss + world_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

    optim = torch.optim.Adam(tensorf.parameters(), lr=3e-2)
    # optim = torch.optim.Adam(tensorf.parameters(), lr=3e-3)
    with torch.enable_grad():
        for _ in tqdm(range(100)):
            ox, oy, oz = torch.rand(3, H, W, H, device=device)/H
            y = (col+oy)
            x = (row+ox)
            z = (line+oz)
            xyz = torch.stack([x, y, z, torch.zeros_like(x)], dim=-1).reshape(-1, 4)
            dist = torch.sqrt(x**2 + y**2 + z**2).reshape(-1)
            mask = (dist < 0.2)
            app_features = tensorf.rf.compute_appfeature(xyz[mask])
            diffuse, tint, roughness, refraction_index, reflectivity, ratio_diffuse = tensorf.diffuse_module(
                xyz[mask], None, app_features)

            shell = (dist[mask] > 0.18)
            full_shell = (dist < 0.2) & (dist > 0.18)
            p_norm = tensorf.normal_module(xyz[mask][shell], app_features[shell])
            gt_norm = xyz[full_shell, :3] / (torch.linalg.norm(xyz[full_shell, :3], dim=1, keepdim=True)+1e-10)
            world_loss = -(p_norm * gt_norm).sum(dim=-1).sum()

            # diffuse = 1
            loss = (diffuse - 1)**2 + (roughness - 0.0)**2 + (refraction_index - 1.4)**2 + (reflectivity - 0.01)**2 + (ratio_diffuse - 0.00)**2
            optim.zero_grad()
            (loss.mean() + world_loss).backward()
            optim.step()

    # init dataset
    dataset = dataset_dict[cfg.dataset.dataset_name]
    test_dataset = dataset(os.path.join(cfg.datadir, cfg.dataset.scenedir), split='test', downsample=cfg.dataset.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = cfg.dataset.ndc_ray

    logfolder = os.path.dirname(cfg.ckpt)
    folder = f'{logfolder}/flat_plane_imgs'
    os.makedirs(folder, exist_ok=True)
    print(f"Saving test to {folder}")
    evaluation(test_dataset,tensorf, cfg, renderer, folder,
               N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
               render_mode=cfg.render_mode)


if __name__ == '__main__':
    main()

