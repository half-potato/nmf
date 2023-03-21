import torch
import utils
import hydra
from omegaconf import DictConfig, OmegaConf
from models.tensor_nerf import TensorNeRF

@hydra.main(version_base=None, config_path='configs', config_name='model_config')
def export_mesh(cfg: DictConfig):
    device = torch.device('cuda')
    ckpt = torch.load(cfg.ckpt, map_location=device)
    tensorf = TensorNeRF.load(ckpt).to(device)

    alpha,_ = tensorf.getDenseAlpha()
    utils.convert_sdf_samples_to_ply(alpha.cpu(), f'{cfg.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)

if __name__ == "__main__":
    export_mesh()
