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
# @torch.no_grad()
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
    # ckpt['config']['bg_module']['bg_resolution'] = ckpt['state_dict']['bg_module.bg_mat'].shape[-1] // 6
    ckpt['config']['bg_module']['bg_resolution'] = 256
    tensorf = hydra.utils.instantiate(cfg.model)(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), grid_size=[128]*3).to(device)
    tensorf2 = TensorNeRF.load(ckpt).to(device)
    # tensorf = TensorNeRF.load(ckpt).to(device)
    tensorf.bg_module = tensorf2.bg_module

    tensorf.rf.set_smoothing(0.5)
    # tensorf.fea2denseAct = 'identity'
    tensorf.fea2denseAct = 'softplus_shift'
    tensorf.max_bounce_rays = 16000
    # tensorf.normal_module = render_modules.AppDimNormal(1, activation=torch.nn.Identity)
    tensorf.normal_module = render_modules.DeepMLPNormal(pospe=16, num_layers=3).to(device)
    tensorf.l = 1
    # tensorf.rf.basis_mat = AddBasis(48)
    tensorf.alphaMask = None
    tensorf.max_recurs = 3
    tensorf.roughness_rays = 5

    # tensorf.rf.init_svd_volume(16)
    # tensorf.to(device)

    # tensorf.rf.density_plane[i][:, :, 40:100, 40:100] += 1
    # tensorf.rf.density_line[i][:, :, ind:ind+8] += 100
    H, W = tensorf.rf.density_plane[0].shape[-2:]
    C = tensorf.rf.density_plane[0].shape[1]
    # tensorf.bg_module.bg_mat[..., :, :, :] = -4
    # tensorf.bg_module.bg_mat[..., 0, :, :] = 0
    # tensorf.bg_module.bg_mat[..., 1, :, :] = 0
    # tensorf.bg_module.bg_mat[..., 2, :, :] = 0

    d = 1
    row, col, line = torch.meshgrid(torch.linspace(-d, d, H, device=device), torch.linspace(-d, d, W, device=device), torch.linspace(-d, d, H, device=device), indexing='ij')

    ord = torch.inf
    ord = 2
    inner_r = 0.13
    outer_r = 0.5
    eps = 0.00

    # train shape
    # optim = torch.optim.Adam(tensorf.parameters(), lr=0.02, betas=(0.9,0.99))
    optim = torch.optim.Adam(tensorf.parameters(), lr=0.02, betas=(0.9,0.99))
    with torch.enable_grad():
        pbar = tqdm(range(250))
        for _ in pbar:
            ox, oy, oz = (torch.rand(3, H, W, H, device=device)*2-1)/H
            y = (col+oy)
            x = (row+ox)
            z = (line+oz)
            xyz = torch.stack([x, y, z, torch.ones_like(x)], dim=-1).reshape(-1, 4)
            inds = np.random.permutation(xyz.shape[0])[:xyz.shape[0]//8]
            # xyz = xyz[inds]
            dist = torch.linalg.norm(xyz[:, :3], dim=1, ord=ord)
            mask = (dist < outer_r)
            # mask2 = (dist > outer_r+eps)
            # mask = (dist < outer_r) & (dist > inner_r) & (y.reshape(-1) > 0)
            feat = tensorf.rf.compute_densityfeature(xyz)
            sigma_feat = tensorf.feature2density(feat)

            # sigma = 1-torch.exp(-sigma_feat * 0.025 * 25)
            sigma = 1-torch.exp(-sigma_feat)
            # sigma = sigma_feat
            loss = ((sigma[mask]-1).abs().mean() + sigma[~mask].abs().mean())
            # loss = (-sigma[mask].clip(max=1).sum() + sigma[~mask].clip(min=1e-8).sum())
            pbar.set_description(f"{loss.detach().item():.06f}")
            optim.zero_grad()
            loss.backward()
            optim.step()

    """
    # train normals
    optim = torch.optim.Adam(tensorf.parameters(), lr=5e-2)
    with torch.enable_grad():
        for _ in tqdm(range(150)):
            ox, oy, oz = torch.rand(3, H, W, H, device=device)/H
            y = (col+oy)
            x = (row+ox)
            z = (line+oz)
            xyz = torch.stack([x, y, z, torch.zeros_like(x)], dim=-1).reshape(-1, 4)
            dist = torch.linalg.norm(xyz[:, :3], dim=1, ord=ord)
            # full_shell = (dist < outer_r + eps) & (dist > inner_r - eps)
            full_shell = (dist < outer_r + eps)

            app_features = tensorf.rf.compute_appfeature(xyz[full_shell])
            p_norm = tensorf.normal_module(xyz[full_shell], app_features)
            sxyz = xyz[full_shell, :3]
            if ord == 2:
                gt_norm = sxyz / (torch.linalg.norm(sxyz, dim=1, keepdim=True)+1e-10)
            else:
                gt_norm = torch.zeros_like(sxyz)
                inds = torch.argmax(sxyz.abs(), dim=1)
                gt_norm[range(sxyz.shape[0]), inds] = torch.sign(sxyz)[range(sxyz.shape[0]), inds]
            world_loss = -(p_norm * gt_norm).sum(dim=-1).sum()
            optim.zero_grad()
            world_loss.backward()
            optim.step()

    r, g, b = 0.955, 0.638, 0.538
    # train appearance
    optim = torch.optim.Adam(tensorf.parameters(), lr=1e-3)
    with torch.enable_grad():
        for _ in tqdm(range(200)):
            ox, oy, oz = torch.rand(3, H, W, H, device=device)/H
            y = (col+oy)
            x = (row+ox)
            z = (line+oz)
            xyz = torch.stack([x, y, z, torch.zeros_like(x)], dim=-1).reshape(-1, 4)
            dist = torch.linalg.norm(xyz[:, :3], dim=1, ord=ord)
            mask = (dist < outer_r + eps)
            app_features = tensorf.rf.compute_appfeature(xyz[mask])
            diffuse, tint, roughness, refraction_index, reflectivity, ratio_diffuse = tensorf.diffuse_module(
                    xyz[mask], torch.rand_like(xyz[..., :3][mask]), app_features)

            # shell = (dist[mask] > inner_r - eps)
            # full_shell = (dist < outer_r + eps) & (dist > inner_r - eps)
            # p_norm = tensorf.normal_module(xyz[mask][shell], app_features[shell])
            # gt_norm = xyz[full_shell, :3] / (torch.linalg.norm(xyz[full_shell, :3], dim=1, keepdim=True)+1e-10)
            # world_loss = -(p_norm * gt_norm).sum(dim=-1).sum()
            
            tint_loss = (tint[..., 0]-r)**2 + (tint[..., 1]-g)**2 + (tint[..., 2]-b)**2
            diffuse_loss = (diffuse[..., 0]-r)**2 + (diffuse[..., 1]-g)**2 + (diffuse[..., 2]-b)**2
            property_loss = (roughness - 0.05)**2 + (refraction_index - 1.5)**2 + (reflectivity - 1.00)**2 + (ratio_diffuse - 0.00)**2
            loss = tint_loss.mean() + diffuse_loss.mean() + property_loss.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
    """

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

