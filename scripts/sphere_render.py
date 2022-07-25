import os
from torch.nn.modules import activation
from tqdm.auto import tqdm
from models.tensor_nerf import TensorNeRF
from models.brdf import PBR, SimplePBR
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from omegaconf import DictConfig, OmegaConf
from models import bg_modules

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

    # init model
    # ckpt = torch.load(cfg.ckpt)
    # # ckpt['config']['bg_module']['bg_resolution'] = ckpt['state_dict']['bg_module.bg_mat'].shape[-1] // 6
    # ckpt['config']['bg_module']['bg_resolution'] = 256
    # del ckpt['state_dict']['diffuse_module.mlp.6.weight']
    # del ckpt['state_dict']['diffuse_module.mlp.6.bias']
    cfg.model.arch.rf.appearance_n_comp = 48
    cfg.model.arch.rf.app_dim = 48
    tensorf = hydra.utils.instantiate(cfg.model.arch)(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), grid_size=[128]*3)
    bg_sd = torch.load('log/mats360_bg.th')
    from models import render_modules
    # bg_module = render_modules.HierarchicalBG(3, render_modules.CubeUnwrap(), bg_resolution=2*1024//4, num_levels=3, featureC=128, num_layers=0)
    bg_module = bg_modules.HierarchicalCubeMap(3, bg_resolution=2048//2**5, num_levels=6, featureC=128, num_layers=0, activation='softplus', power=2)
    # bg_module = render_modules.BackgroundRender(3, render_modules.PanoUnwrap(), bg_resolution=2*1024, featureC=128, num_layers=0)
    bg_module.load_state_dict(bg_sd)
    tensorf.bg_module = bg_module
    tensorf.brdf = SimplePBR(0)
    tensorf.rf.set_smoothing(1.5)
    tensorf = tensorf.to(device)
    ic(tensorf)

    H, W = tensorf.rf.density_plane[0].shape[-2:]
    C = tensorf.rf.density_plane[0].shape[1]

    d = 1
    row, col, line = torch.meshgrid(torch.linspace(-d, d, H, device=device), torch.linspace(-d, d, W, device=device), torch.linspace(-d, d, H, device=device), indexing='ij')
    grid = torch.stack([row, col, line, torch.ones_like(row)], dim=-1).reshape(1, -1, 4)
    N = grid.shape[1]

    ord = torch.inf
    ord = 2
    inner_r = 0.13
    outer_r = 0.25
    eps = 0.00
    offset = torch.tensor([0.0, 0, 0.70]).to(device).reshape(1, 3)

    def gen_mask(xyz, return_all=False):
        dist = torch.linalg.norm(xyz[:, :3]-offset, dim=1, ord=2)
        dist2 = torch.linalg.norm(xyz[:, :3], dim=1, ord=ord)
        mask1 = (dist < outer_r)
        mask2 = (dist2 < 0.25) 
        if return_all:
            # return mask1 & ~mask2, mask2 & ~mask1
            return mask1, mask2
        else:
            return mask1 | mask2


    # train shape
    optim = torch.optim.Adam(tensorf.parameters(), lr=0.1, betas=(0.9,0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=500)
    with torch.enable_grad():
        # TODO REMOVE
        pbar = tqdm(range(500))
        for _ in pbar:
            noise = torch.rand(1, N, 4, device=device)/H
            noise[..., -1] = 0
            xyz = (grid + noise).reshape(-1, 4)
            # inds = np.random.permutation(xyz.shape[0])[:xyz.shape[0]//2]
            mask = gen_mask(xyz[:, :3])
            # mask2 = (dist > outer_r+eps)
            # mask = (dist < outer_r) & (dist > inner_r) & (y.reshape(-1) > 0)
            feat = tensorf.rf.compute_densityfeature(xyz)
            sigma_feat = tensorf.feature2density(feat)

            # sigma = 1-torch.exp(-sigma_feat * 0.025 * 25)
            sigma = 1-torch.exp(-sigma_feat)
            # sigma = sigma_feat
            loss = (sigma[mask]-1).abs().mean() + sigma[~mask].abs().mean()
            # loss = (-sigma[mask].clip(max=1).sum() + sigma[~mask].clip(min=1e-8).sum())
            pbar.set_description(f"Shape loss: {loss.detach().item():.06f} LR: {scheduler.get_last_lr()}")
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()


    # rgb = torch.tensor([0.955, 0.638, 0.538], device=device).reshape(1, 3)
    # rgb = torch.tensor([0.98, 0.82, 0.76], device=device).reshape(1, 3)
    rgb = torch.tensor([0.98, 0.539, 0.316], device=device).reshape(1, 3)
    # r, g, b = 1, 1, 1
    # train normals
    # f0_col = torch.tensor([0.955, 0.638, 0.538]).to(device)
    # optim = torch.optim.Adam(tensorf.parameters(), lr=0.0100)
    optim = torch.optim.Adam(tensorf.parameters(), lr=0.0025)
    # optim = torch.optim.Adam(tensorf.parameters(), lr=0.0010)
    # optim = torch.optim.RMSprop(tensorf.parameters(), lr=0.0500)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
    with torch.enable_grad():
        pbar = tqdm(range(1000))
        for _ in pbar:
            noise = torch.rand(1, N, 4, device=device)/H
            noise[..., -1] = 0
            xyz = (grid + noise).reshape(-1, 4)

            full_shell = gen_mask(xyz[:, :3])
            xyz = xyz[full_shell]
            # inds = np.random.permutation(xyz.shape[0])[:xyz.shape[0]//2]
            # xyz = xyz[inds]

            """
            with torch.no_grad():
                gt_norms = tensorf.calculate_normals(xyz).detach()

            zero_mask = torch.linalg.norm(gt_norms, dim=-1) > 0.1
            xyz = xyz[zero_mask]
            gt_norms = gt_norms[zero_mask]
            """
            full_shell = gen_mask(xyz[:, :3])
            sphere_mask, cube_mask = gen_mask(xyz[:, :3], return_all=True)
            sxyz = xyz[sphere_mask, :3]-offset
            cxyz = xyz[cube_mask, :3]

            sgt_norm = sxyz / (torch.linalg.norm(sxyz, dim=1, keepdim=True)+1e-10)

            if ord == 2:
                cgt_norm = cxyz / (torch.linalg.norm(cxyz, dim=1, keepdim=True)+1e-10)
            else:
                cgt_norm = torch.zeros_like(cxyz)
                inds = torch.argmax(cxyz.abs(), dim=1)
                cgt_norm[range(cxyz.shape[0]), inds] = torch.sign(cxyz)[range(cxyz.shape[0]), inds]

            gt_norms = torch.zeros_like(xyz[:, :3])
            gt_norms[sphere_mask] = sgt_norm
            gt_norms[cube_mask] = cgt_norm
            gt_norms = gt_norms[full_shell]
            # """
            app_features = tensorf.rf.compute_appfeature(xyz)

            # app_features = (app_features + torch.randn_like(app_features) * tensorf.appdim_noise_std)

            p_norm = tensorf.normal_module(xyz, app_features)

            # norm_diff = (1-(p_norm * gt_norms).sum(dim=-1))
            norm_diff_l2 = torch.linalg.norm(p_norm - gt_norms, dim=-1)
            world_loss = (norm_diff_l2).mean()

            # appearance
            diffuse, tint, matprop = tensorf.diffuse_module(
                    xyz[full_shell], torch.rand_like(xyz[..., :3][full_shell]), app_features[full_shell])

            # shell = (dist[mask] > inner_r - eps)
            # full_shell = (dist < outer_r + eps) & (dist > inner_r - eps)
            # p_norm = tensorf.normal_module(xyz[mask][shell], app_features[shell])
            # gt_norm = xyz[full_shell, :3] / (torch.linalg.norm(xyz[full_shell, :3], dim=1, keepdim=True)+1e-10)
            # world_loss = -(p_norm * gt_norm).sum(dim=-1).sum()
            
            tint_loss = ((tint-rgb)**2).sum()
            diffuse_loss = (diffuse[..., 0]-0)**2 + (diffuse[..., 1]-0)**2 + (diffuse[..., 2]-0)**2
            property_loss = (matprop['refraction_index'] - 1.5)**2 + (matprop['reflectivity'] - 0.00)**2 + (matprop['ratio_diffuse'] - 0.10)**2 + (matprop['ambient'] + 0.1)**2 + \
                            (matprop['roughness'] - 0.5)**2
            app_loss = tint_loss.mean() + diffuse_loss.mean() + property_loss.mean()
            loss = 1e-4*app_loss + world_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_description(f"World loss: {world_loss.item():.07f} Loss: {loss.item():.07f} LR: {scheduler.get_last_lr()}")
            scheduler.step()


    # init dataset
    dataset = dataset_dict[cfg.dataset.dataset_name]
    test_dataset = dataset(os.path.join(cfg.datadir, cfg.dataset.scenedir), split='train', downsample=cfg.dataset.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = cfg.dataset.ndc_ray

    folder = f'log/flat_plane_imgs'
    os.makedirs(folder, exist_ok=True)
    print(f"Saving test to {folder}")
    evaluation(test_dataset,tensorf, cfg, renderer, folder,
               N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
               render_mode=cfg.render_mode)


if __name__ == '__main__':
    main()

