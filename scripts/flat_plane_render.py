import os
from tqdm.auto import tqdm
from models.tensor_nerf import TensorNeRF
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from omegaconf import DictConfig, OmegaConf

from dataLoader import dataset_dict
import sys
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import math

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
    tensorf.rf.set_smoothing(2)

    # model surgery
    # zero out values
    for density_line in tensorf.rf.density_line:
        density_line *= 0
    for density_plane in tensorf.rf.density_plane:
        density_plane *= 0
    for app_line in tensorf.rf.app_line:
        app_line *= 0
    for app_plane in tensorf.rf.app_plane:
        app_plane *= 0
    # set density to plane

    i = 0
    n = [0, 0, 0]
    n[i] = 1
    # tensorf.rf.density_plane[i][:, :, 40:100, 40:100] += 1
    # tensorf.rf.density_line[i][:, :, ind:ind+8] += 100
    tensorf.rf.app_plane[i][:, 8] = 0
    tensorf.rf.app_plane[i][:, 9] = -1
    tensorf.rf.app_plane[i][:, 10] = 0
    tensorf.rf.app_line[i] += 1
    H, W = tensorf.rf.density_plane[0].shape[-2:]
    C = tensorf.rf.density_plane[0].shape[1]

    d = 10
    k = d-1
    row, col, line = torch.meshgrid(torch.linspace(-d-k, d-k, H, device=device), torch.linspace(-d+0, d+0, W, device=device), torch.linspace(1, 0, H, device=device), indexing='ij')

    c = H // C
    l = 1
    o = 150
    dist = torch.sqrt(row**2 + col**2 + line**2)
    mask = (dist < 1).float()
    for j in range(C):
        m = mask[..., c*j:c*j+c].sum(dim=-1) > 0
        tensorf.rf.density_plane[i][0, j][m] = 10

        k = C - j
        p = l * C + o - 1
        tensorf.rf.density_line[i][0, j, l*j+o:l*j+l+o] += 10
        tensorf.rf.density_line[i][0, j, l*k+p:l*k+l+p] += 10

    optim = torch.optim.Adam(tensorf.parameters(), lr=1e-3)
    with torch.enable_grad():
        for _ in tqdm(range(50)):
            ox, oy, oz = torch.rand(3, H, W, H, device=device)
            y = (col+oy)
            x = (row+ox)
            z = (line+oz)
            xyz = torch.stack([x, y, z], dim=-1)
            dist = torch.sqrt(x**2 + y**2 + z**2)
            mask = (dist < 0.3).float()
            feat = tensorf.rf.compute_densityfeature(xyz)
            sigma_feat = tensorf.feature2density(feat)
            sigma = 1-torch.exp(-sigma_feat * 0.025)
            ic(sigma.min(), sigma.max(), sigma_feat.min(), sigma_feat.max(), feat.min(), feat.max())
            # loss = ((sigma - mask)**2).mean()
            loss = ((feat - 200*mask)**2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
    # print("HI")

    tensorf.rf.basis_mat = AddBasis(48)
    # tensorf.rf.basis_mat = torch.nn.Identity()
    # set hue and diffuse to white
    tensorf.rf.app_plane[i][:, :6] += 2
    # set roughness to 0
    tensorf.rf.app_plane[i][:, 6] -= 4

    ior = 1.4 - 1
    tensorf.rf.app_plane[i][:, 7] = math.log(ior/(1-ior))

    reflectivity = 0.01
    tensorf.rf.app_plane[i][:, 8] = math.log(reflectivity/(1-reflectivity))
    # diffuse_ratio = 0.99
    diffuse_ratio = 0.01
    tensorf.rf.app_plane[i][:, 9] = math.log(diffuse_ratio/(1-diffuse_ratio))


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
