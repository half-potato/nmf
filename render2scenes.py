from pathlib import Path
import sys
import os
base_path = Path(os.path.abspath('')).parent
print(base_path)
sys.path.append(str(base_path))
import torch
from modules.tensor_nerf import TensorNeRF
from fields.listrf import ListRF
from samplers.nerf_acc import NerfAccSampler
from icecream import ic

import imageio

log_dir = Path("log") / "noprednorms_nl0_conserve_pb0"
ckpt1 = log_dir / 'car_v38' / 'car_v38.th'
ckpt2 = log_dir / 'toaster_v38' / 'toaster_v38.th'

tensorf1 = TensorNeRF.load(torch.load(ckpt1), near_far=[2, 6], strict=False)
tensorf2 = TensorNeRF.load(torch.load(ckpt2), near_far=[2, 6], strict=False)
device = torch.device('cuda')
tensorf1 = tensorf1.to(device)
tensorf2 = tensorf2.to(device)
offsets = [
    torch.tensor([[0, 0.0, 0, 0]], device=device),
    torch.tensor([[0.5, 0.0, 0, 0]], device=device),
]

new_abbs = [
    torch.tensor([[-0.5,-0.5,-0.5],
                  [ 0.5, 0.5, 0.5]], device=device)*0.1,
    torch.tensor([[-0.5,-0.5,-0.5],
                  [ 0.5, 0.5, 0.5]], device=device)*0.1,
]

# new_abbs = [
#     torch.tensor([[-1.5,-1.5,-1.5],
#                   [ 1.5, 1.5, 1.5]], device=device),
#     torch.tensor([[-1.5,-1.5,-1.5],
#                   [ 1.5, 1.5, 1.5]], device=device),
# ]

listrf = ListRF([tensorf1.rf, tensorf2.rf], offsets)
tensorf2.rf = listrf

aabb1 = tensorf1.rf.aabb + offsets[0][:, :3]
aabb2 = tensorf1.rf.aabb + offsets[0][:, :3]
aabb = torch.stack([
    torch.minimum(aabb1[0], aabb1[0]),
    torch.maximum(aabb1[1], aabb1[1]),
], dim=0)

tensorf2.sampler = NerfAccSampler(aabb, [2, 5], grid_size=128,
         render_n_samples=1024, max_samples=-1, multiplier=1,
         test_multiplier=1, update_freq=16, shrink_iters=[],
         alpha_thres = 1e-4, ema_decay = 0.95, occ_thre=0.01,
         warmup_iters=256).to(device)
tensorf2.sampler.update(tensorf2.rf, init=True)
for i in range(1000):
    tensorf2.sampler.check_schedule(i, 1, tensorf2.rf)

from dataLoader.ray_utils import get_ray_directions, get_rays
from renderer import chunk_renderer, BundleRender
import numpy as np

w = 200
h = 200
camera_angle_x = 0.6194058656692505

fx = 0.5 * w / np.tan(0.5 * camera_angle_x)  # original focal length
fy = fx

directions = get_ray_directions(h, w, [fx,fy])  # (h, w, 3)
directions = directions / torch.norm(directions, dim=-1, keepdim=True)
c2w = torch.eye(4)
blender2opencv = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).float()

c2w = torch.tensor([
    [
        -0.9999021291732788,
        0.004192245192825794,
        -0.013345719315111637,
        -0.05379832163453102
    ],
    [
        -0.013988680206239223,
        -0.2996590733528137,
        0.95394366979599,
        3.845470428466797
    ],
    [
        -4.6566125955216364e-10,
        0.9540371894836426,
        0.29968830943107605,
        1.2080823183059692
    ],
    [
        0.0,
        0.0,
        0.0,
        1.0
    ]
]) @ blender2opencv

c2w = torch.tensor([
    [
        -0.9999999403953552,
        0.0,
        0.0,
        0.0
    ],
    [
        0.0,
        -0.7341099977493286,
        0.6790305972099304,
        2.737260103225708
    ],
    [
        0.0,
        0.6790306568145752,
        0.7341098785400391,
        2.959291696548462
    ],
    [
        0.0,
        0.0,
        0.0,
        1.0
    ]
]) @ blender2opencv


rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)

rays = torch.cat([rays_o, rays_d], 1).to(device)

tensorf2.model.max_retrace_rays=[10000]
tensorf2.model.max_brdf_rays=[500000, 500000]
tensorf2.eval_batch_size=500
ic(tensorf2.sampler.max_samples)

brender = BundleRender(chunk_renderer, h, w, 1)
ims, stats = brender(rays, tensorf2, N_samples=-1, ndc_ray=False, is_train=False)
imageio.imwrite('2scene.png', (ims['rgb_map'].clip(0, 1)*255).numpy().astype(np.uint8))

