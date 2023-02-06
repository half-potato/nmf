from collections import defaultdict
import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
from modules import tonemap
from loguru import logger

import torch.nn.functional as F
import matplotlib.pyplot as plt
from icecream import ic
from modules.tensor_nerf import LOGGER
import traceback
from pathlib import Path
import yaml
from sklearn import linear_model

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def stack_tensors(data):
    # stack it boyyy
    for key in data.keys():
        try:
            if len(data[key]) == 1:
                data[key] = data[key][0]
                continue
            if torch.is_tensor(data[key][0]) and len(data[key][0].shape) > 0:
                data[key] = torch.cat(data[key], dim=0)
            else:
                data[key] = torch.tensor(data[key])
        except:
            pass
        #     traceback.print_exc()
        #     ic(key, data[key][0])
    return data

def chunk_renderer(rays, tensorf, focal, keys=['rgb_map'], chunk=4096, render2completion=False, **kwargs):

    ims = defaultdict(list)
    stats = defaultdict(list)
    N_rays_all = rays.shape[0]
    rng = range(N_rays_all // chunk + int(N_rays_all % chunk > 0))
    # if render2completion:
    #     rng = tqdm(rng)
    for chunk_idx in rng:
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk]#.to(device)
        if rays_chunk.numel() == 0:
            continue
        need_rendering = torch.ones((rays_chunk.shape[0]), dtype=bool, device=rays_chunk.device)
        while need_rendering.sum() > 0:
            rays_p = rays_chunk[need_rendering]
            if rays_p.shape[0] == 0:
                break
            cims, cstats = tensorf(rays_p, focal, **kwargs)
            # collect stuff in keys if specified, else collect everything
            if keys is not None:
                for key in keys:
                    if key in cims:
                        ims[key].append(cims[key])
                    if key in cstats:
                        stats[key].append(cstats[key])
            else:
                for key in cims.keys():
                    if key in cims:
                        ims[key].append(cims[key])
                for key in cstats.keys():
                    if key in cstats:
                        stats[key].append(cstats[key])

            whole_valid = cstats['whole_valid']
            if not render2completion:
                break
            need_rendering[need_rendering.clone()] = ~whole_valid

    return stack_tensors(ims), stack_tensors(stats)

class BundleRender:
    def __init__(self, base_renderer, H, W, focal, bundle_size=1, scale_normal=False):
        self.base_renderer = base_renderer
        self.bundle_size = bundle_size
        self.H = H 
        self.W = W
        self.scale_normal = scale_normal
        self.focal = focal

    @torch.no_grad()
    def __call__(self, rays, tensorf, **kwargs):
        height, width = self.H, self.W
        fH = height
        fW = width
        device = rays.device

        LOGGER.reset()
        ims, stats = self.base_renderer(
            rays, tensorf, keys=None,
            focal=self.focal, chunk=tensorf.eval_batch_size, render2completion=True, **kwargs)

        LOGGER.save('rays.pkl')
        LOGGER.reset()
        points = ims['termination_xyz']
        point = points[len(points)//2].to(device)

        if hasattr(tensorf.model, 'recover_envmap') and False:
            env_map = tensorf.model.recover_envmap(512, xyz=point, roughness=0.01)
            env_map = (env_map.detach().cpu().numpy() * 255).astype('uint8')
            # col_map = (col_map.detach().cpu().numpy() * 255).astype('uint8')
            vals = dict(
                env_map=env_map,
                # col_map=col_map,
            )
        else:
            vals = {}

        def reshape(val_map):
            val_map = val_map.reshape((height, width, -1))
            # val_map = val_map.reshape((fH, fW, -1))[:self.H, :self.W, :]
            return val_map

        return dotdict(
            **{k: reshape(ims[k].detach()).cpu() for k in ims.keys()},
            **vals,
        ), stats


def depth_to_normals(depth, focal):
    """Assuming `depth` is orthographic, linearize it to a set of normals."""

    f_blur = torch.tensor([1, 2, 1]) / 4
    f_edge = torch.tensor([-1, 0, 1]) / 2
    depth = depth.unsqueeze(0).unsqueeze(0)
    dy = F.conv2d(depth, (f_blur[None, :] * f_edge[:, None]).unsqueeze(0).unsqueeze(0), padding=1)[0, 0]
    dx = F.conv2d(depth, (f_blur[:, None] * f_edge[None, :]).unsqueeze(0).unsqueeze(0), padding=1)[0, 0]

    # so dx, dy are in image space but we want to transform them to world space
    dx = dx * focal * 2 / depth[0, 0]
    dy = dy * focal * 2 / depth[0, 0]
    inv_denom = 1 / torch.sqrt(1 + dx**2 + dy**2)
    normals = torch.stack([dx * inv_denom, -dy * inv_denom, inv_denom], -1)
    return normals

def evaluate(iterator, test_dataset,tensorf, renderer, savePath=None, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', bundle_size=1, gt_bg=None):
    print("Eval")
    PSNRs, rgb_maps, depth_maps = [], [], []
    norm_errs = []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+"/normal", exist_ok=True)
    os.makedirs(savePath+"/world_normal", exist_ok=True)
    os.makedirs(savePath+"/normal_err", exist_ok=True)
    os.makedirs(savePath+"/err", exist_ok=True)
    os.makedirs(savePath+"/surf_width", exist_ok=True)
    # os.makedirs(savePath+"/debug", exist_ok=True)
    os.makedirs(savePath+"/tint", exist_ok=True)
    os.makedirs(savePath+"/spec", exist_ok=True)
    # os.makedirs(savePath+"/brdf", exist_ok=True)
    os.makedirs(savePath+"/diffuse", exist_ok=True)
    os.makedirs(savePath+"/roughness", exist_ok=True)
    # os.makedirs(savePath+"/r0", exist_ok=True)
    # os.makedirs(savePath+"/transmitted", exist_ok=True)
    # os.makedirs(savePath+"/diffuse_light", exist_ok=True)
    os.makedirs(savePath+"/cross_section", exist_ok=True)
    # os.makedirs(savePath+"/envmaps", exist_ok=True)


    # save brdf stuff
    if hasattr(tensorf.model, 'graph_brdfs'):
        N = 8
        n = 0
        while n < N:
            xyz = torch.rand(200, 4, device=device)*2-1
            xyz[:, 3] *= 0
            sigma_feat = tensorf.rf.compute_densityfeature(xyz)
            xyz = xyz[sigma_feat > sigma_feat.mean()][:8]
            n = xyz.shape[0]
        feat = tensorf.rf.compute_appfeature(xyz)
        viewangs = torch.linspace(0, np.pi, 8, device=device)
        viewdirs = torch.stack([
            torch.cos(viewangs),
            torch.zeros_like(viewangs),
            -torch.sin(viewangs),
        ], dim=-1).reshape(-1, 3).to(device)
        res = 100
        brdf_im = tensorf.model.graph_brdfs(xyz, viewdirs, feat, res).cpu()
        bg_path = Path(savePath) / 'brdf'
        bg_path.mkdir(exist_ok=True, parents=True)
        imageio.imwrite(bg_path / f'{prtx}brdf_map.exr', brdf_im)


    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh
    focal = (test_dataset.focal[0] if ndc_ray else test_dataset.fx)
    brender = BundleRender(renderer, H, W, focal)

    # if tensorf.ref_module is not None:
    #     os.makedirs(savePath+"/envmaps", exist_ok=True)
    #     env_map, col_map = tensorf.recover_envmap(512, xyz=torch.tensor([-0.3042,  0.8466,  0.8462,  0.0027], device='cuda:0'))
    #     env_map = (env_map.clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
    #     col_map = (col_map.clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
    #     imageio.imwrite(f'{savePath}/envmaps/{prtx}view_map.png', col_map)
    #     imageio.imwrite(f'{savePath}/envmaps/{prtx}ref_map.png', env_map)
    if tensorf.model.visibility_module is not None:
        os.makedirs(savePath+"/viscache", exist_ok=True)
        tensorf.model.visibility_module.save(f'{savePath}/viscache/', prtx)

    T2 = torch.tensor([
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    T = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    tensorf.eval()
    tint_psnrs = []
    for idx, im_idx, rays, gt_rgb in iterator():

        ims, stats = brender(rays, tensorf, N_samples=N_samples, ndc_ray=ndc_ray, is_train=False)

        # H, W, _ = normal.shape
        # normal = normal.reshape(-1, 3)# @ pose[:3, :3]
        # normal = normal.reshape(H, W, 3)
        # bottom of the sphere is green
        # top is blue
        vis_normal = (ims.normal * 127 + 128).clamp(0, 255).byte()
        vis_world_normal = (ims.world_normal * 127 + 128).clamp(0, 255).byte()
        # vis_normal = (normal * 255).clamp(0, 255).byte()

        err_map = (ims.rgb_map.clip(0, 1) - gt_rgb.clip(0, 1)) + 0.5

        vis_depth_map, _ = visualize_depth_numpy(ims.depth.numpy(),near_far)

        mask = ims.acc_map.reshape(-1) > 0.1
        try:
            gt_tint = test_dataset.get_tint(im_idx)
            Y = (gt_tint.reshape(-1, 3)[mask]).numpy()
            X = (ims.tint.reshape(-1, 3)[mask]).numpy()
            model = linear_model.LinearRegression()
            model.fit(X, Y)
            pred_Y = model.predict(X)

            mean_tint_err = ((pred_Y-Y)**2).mean()
            tint_psnrs.append(-10.0 * np.log(mean_tint_err.item()) / np.log(10.0))
        except:
            pass
        # try:
        #     gt_tint = test_dataset.get_tint(im_idx)
        #     mean_tint_err = ((gt_tint-ims.tint)**2).mean()
        #     tint_psnrs.append(-10.0 * np.log(mean_tint_err.item()) / np.log(10.0))
        # except:
        #     pass
        if gt_rgb is not None:
            try:
                gt_normal = test_dataset.get_normal(im_idx)
                # vis_gt_normal = (gt_normal * 127 + 128).clamp(0, 255).byte()
                # X = normal.reshape(-1, 3)
                # Y = gt_normal.reshape(-1, 3)
                # u, d, vh = torch.linalg.svd(X.T @ Y)
                # ic(u @ vh)
                # mask = (gt_normal[..., 0] == 1) & (gt_normal[..., 1] == 1) & (gt_normal[..., 2] == 1)
                # gt_normal[mask] = 0
                norm_err = torch.arccos((ims.normal * gt_normal).sum(dim=-1).clip(min=1e-8, max=1-1e-8)) * 180/np.pi
                norm_err[torch.isnan(norm_err)] = 0
                norm_err *= ims.acc_map.squeeze(-1)
                norm_errs.append(norm_err.mean())
                if savePath is not None:
                    imageio.imwrite(f'{savePath}/normal_err/{prtx}{idx:03d}.exr', norm_err.numpy())
                    # imageio.imwrite(f'{savePath}/normal_err/{prtx}{idx:03d}.png', vis_gt_normal)
            except:
                pass
                # traceback.print_exc()
            loss = torch.mean((ims.rgb_map.clip(0, 1) - gt_rgb.clip(0, 1)) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            # fig, axs = plt.subplots(2, 2)
            # axs[0, 0].imshow(rgb_map)
            # axs[1, 0].imshow(gt_rgb)
            # axs[0, 1].imshow(rgb_map-gt_rgb)
            # axs[1, 1].imshow(depth_map)
            # plt.show()

            if compute_extra_metrics:
                ssim = rgb_ssim(ims.rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), ims.rgb_map.numpy(), 'alex', device)
                l_v = rgb_lpips(gt_rgb.numpy(), ims.rgb_map.numpy(), 'vgg', device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (ims.rgb_map.clamp(0, 1).numpy() * 255).astype('uint8')
        err_map = (err_map.clamp(0, 1).numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(vis_depth_map)

        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, vis_depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.exr', ims.depth.numpy())
            imageio.imwrite(f'{savePath}/normal/{prtx}{idx:03d}.png', vis_normal)
            if 'spec' in ims:
                imageio.imwrite(f'{savePath}/spec/{prtx}{idx:03d}.exr', ims.spec.numpy())
            if 'roughness' in ims:
                imageio.imwrite(f'{savePath}/roughness/{prtx}{idx:03d}.exr', ims.roughness)
            if 'tint' in ims:
                imageio.imwrite(f'{savePath}/tint/{prtx}{idx:03d}.exr', ims.tint.numpy())
            if 'diffuse' in ims:
                imageio.imwrite(f'{savePath}/diffuse/{prtx}{idx:03d}.png', (255*ims.diffuse.clamp(0, 1).numpy()).astype(np.uint8))
            imageio.imwrite(f'{savePath}/world_normal/{prtx}{idx:03d}.png', vis_world_normal)
            imageio.imwrite(f'{savePath}/err/{prtx}{idx:03d}.png', err_map)
            imageio.imwrite(f'{savePath}/surf_width/{prtx}{idx:03d}.png', ims.surf_width.numpy().astype(np.uint8))

            cross_section = (ims.cross_section.clamp(0, 1).numpy() * 255).astype('uint8')
            imageio.imwrite(f'{savePath}/cross_section/{prtx}{idx:03d}.png', cross_section)
            # debug = 255*data.debug_map.clamp(0, 1)
            if 'env_map' in ims:
                imageio.imwrite(f'{savePath}/envmaps/{prtx}ref_map_{idx:03d}.png', ims.env_map)
                # imageio.imwrite(f'{savePath}/envmaps/{prtx}view_map_{idx:03d}.png', data.col_map)

    tensorf.train()
    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=10, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=10, quality=10)
    # for i in range(100):
    #     env_map, col_map = tensorf.recover_envmap(1024)
    #     # plt.imshow(col_map.cpu())
    #     # plt.figure()
    #     # plt.imshow(env_map.cpu())
    #     # plt.show()
    #     env_map = (env_map.cpu().numpy() * 255).astype('uint8')
    #     col_map = (col_map.cpu().numpy() * 255).astype('uint8')
    #     imageio.imwrite(f'{savePath}/{prtx}col_map{i}.png', col_map)
    #     imageio.imwrite(f'{savePath}/{prtx}env_map{i}.png', env_map)
    final_stats = {}
    if tint_psnrs:

        final_stats['tint_psnr'] = float(np.mean(np.asarray(tint_psnrs)))

    if tensorf.bg_module is not None:
        tm = tonemap.HDRTonemap()
        bg_path = Path(savePath) / 'envmaps'
        bg_path.mkdir(exist_ok=True, parents=True)
        tensorf.bg_module.save(bg_path, prefix=prtx, tonemap=tm)

        if gt_bg is not None:
            bg_psnr = tensorf.bg_module.calc_envmap_psnr(gt_bg)
            logger.info(f'bg_psnr={float(bg_psnr):.3f}')
            final_stats['bg_psnr'] = float(bg_psnr)


    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        final_stats['psnr'] = psnr.item()
        if len(norm_errs) > 0:
            norm_err = float(np.mean(np.asarray(norm_errs)))
        else:
            norm_err = 0
        print(f"Norm err: {norm_err}")
        final_stats['norm_err'] = norm_err
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))

            final_stats['ssim'] = float(ssim)
            final_stats['l_alex'] = float(l_a)
            final_stats['l_vgg'] = float(l_v)
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))
    with open(f'{savePath}/stats{prtx}.yaml', 'w') as f:
        yaml.dump(final_stats, f)


    return dict(psnrs=PSNRs, norm_errs=norm_errs)

# @torch.no_grad()
def evaluation(test_dataset,tensorf, unused, renderer, *args, N_vis=5, device='cuda', **kwargs):

    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    W, H = test_dataset.img_wh
    def iterator():
        for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

            rays = samples.view(-1,samples.shape[-1]).to(device)
            if len(test_dataset.all_rgbs):
                gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
                yield idx, idxs[idx], rays, gt_rgb
            else:
                yield idx, idxs[idx], rays, None
    return evaluate(iterator, test_dataset, tensorf, renderer, *args, device=device, **kwargs)

# @torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, *args, device='cuda', ndc_ray=False, **kwargs):
    W, H = test_dataset.img_wh
    def iterator():
        for idx, c2w in tqdm(enumerate(c2ws)):

            c2w = torch.FloatTensor(c2w)
            rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
            if ndc_ray:
                rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            yield idx, idx, rays, None
    return evaluate(iterator, test_dataset, tensorf, renderer, *args, ndc_ray=ndc_ray, **kwargs)
