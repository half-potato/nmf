from collections import defaultdict
import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from utils import *
from dataLoader.ray_utils import ndc_rays_blender

import torch.nn.functional as F
import matplotlib.pyplot as plt
from icecream import ic

def chunk_renderer(rays, tensorf, focal, keys=['rgb_map'], chunk=4096, **kwargs):

    data = defaultdict(list)
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk]#.to(device)
    
        cdata = tensorf(rays_chunk, focal, **kwargs)
        for key in keys:
            data[key].append(cdata[key])

    # stack it boyyy
    for key in keys:
        if len(data[key]) == 1:
            data[key] = data[key][0]
            continue
        if torch.is_tensor(data[key][0]):
            data[key] = torch.cat(data[key], dim=0)
        else:
            data[key] = torch.tensor(data[key])
    return data

class BundleRender:
    def __init__(self, base_renderer, render_mode, H, W, focal, bundle_size=1, chunk=2*512, scale_normal=False):
        self.render_mode = render_mode
        self.base_renderer = base_renderer
        self.bundle_size = bundle_size
        self.H = H 
        self.W = W
        self.scale_normal = scale_normal
        self.focal = focal
        self.chunk = chunk

    @torch.no_grad()
    def __call__(self, rays, tensorf, **kwargs):
        height, width = self.H, self.W
        ray_dim = rays.shape[-1]
        if self.render_mode == 'decimate':
            rays = rays.reshape(self.H, self.W, ray_dim)
            rays = rays[::self.bundle_size, ::self.bundle_size]
            height = rays.shape[0]
            width = rays.shape[1]
            rays = rays.reshape(-1, ray_dim)
            fH = height * self.bundle_size
            fW = width * self.bundle_size
        else:
            fH = height
            fW = width
        num_rays = height * width
        device = rays.device

        data = self.base_renderer(rays, tensorf, keys=['depth_map', 'rgb_map', 'normal_map', 'acc_map', 'termination_xyz', 'debug_map'],
                                  focal=self.focal, chunk=self.chunk, **kwargs)
        rgb_map = data['rgb_map']
        depth_map = data['depth_map']
        normal_map = data['normal_map']
        debug_map = data['debug_map']
        acc_map = data['acc_map']
        points = data['termination_xyz']
        #  ind = [598,532]
        ind = [height // 2, width // 2]
        point = points.reshape(height, width, -1)[ind[0], ind[1]].to(device)

        env_map, col_map = tensorf.recover_envmap(512, xyz=point)
        env_map = (env_map.detach().cpu().numpy() * 255).astype('uint8')
        col_map = (col_map.detach().cpu().numpy() * 255).astype('uint8')

        if self.render_mode == 'decimate':
            # plt.imshow(normal_map.reshape(height, width, 3).cpu())
            # plt.figure()
            normal_map = None

        def reshape(val_map):
            val_map = val_map.reshape((height, width, self.bundle_size, self.bundle_size, -1))
            val_map = val_map.permute((0, 2, 1, 3, 4))
            if self.render_mode == 'center':
                val_map = val_map[:, self.bundle_size//2, :, self.bundle_size//2]
            elif self.render_mode == 'mean':
                print("Mean is incorrect. It should not be done this way. It should take averages on pixels")
                val_map = val_map.mean(axis=1, keepdim=True).mean(axis=3, keepdim=True)
            val_map = val_map.reshape((fH, fW, -1))[:self.H, :self.W, :]
            return val_map


        rgb_map, depth_map, acc_map = reshape(rgb_map.detach()).cpu(), reshape(depth_map.detach()).cpu(), reshape(acc_map.detach()).cpu()
        debug_map = reshape(debug_map.detach()).cpu()
        rgb_map = rgb_map.clamp(0.0, 1.0)
        if normal_map is not None:
            normal_map = normal_map.reshape(height, width, 3).cpu()
        else:
            print(f"Falling back to normals from depth map. Render mode: {self.render_mode}")
            normal_map = depth_to_normals(depth_map, self.focal)

        # normal_map = acc_map * normal_map + (1-acc_map) * 0
        # plt.imshow(normal_map/2+0.5)
        # plt.figure()

        # normal_map = depth_to_normals(depth_map, self.focal)
        # normal_map = acc_map * normal_map + (1-acc_map) * 0
        # plt.imshow(acc_map)
        # plt.figure()
        # plt.imshow(depth_map)
        # plt.figure()
        # plt.imshow(rgb_map)
        # plt.figure()
        # plt.imshow(normal_map/2+0.5)
        # plt.show()
        # points = points.cpu()
        # ic(points.shape, acc_map.shape)
        # mask = acc_map.flatten() > 0.1
        # fig = go.Figure(data=go.Cone(
        #     x=points[mask, 0],
        #     y=points[mask, 1],
        #     z=points[mask, 2],
        #     # u=normal_map.reshape(-1, 3)[mask, 0],
        #     # v=normal_map.reshape(-1, 3)[mask, 2],
        #     # w=normal_map.reshape(-1, 3)[mask, 1],
        #     u=normal_map.reshape(-1, 3)[mask, 0],
        #     v=normal_map.reshape(-1, 3)[mask, 1],
        #     w=normal_map.reshape(-1, 3)[mask, 2],
        # ))
        # fig.show()
        # assert(False)

        return rgb_map, depth_map, debug_map, normal_map, env_map, col_map


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
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', bundle_size=1, render_mode='decimate'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+"/envmaps", exist_ok=True)
    ic(N_samples)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh
    focal = (test_dataset.focal[0] if ndc_ray else test_dataset.focal)
    brender = BundleRender(renderer, render_mode, H, W, focal)
    env_map, col_map = tensorf.recover_envmap(512, xyz=torch.tensor([-0.3042,  0.8466,  0.8462,  0.0027], device='cuda:0'))
    env_map = (env_map.detach().cpu().numpy() * 255).astype('uint8')
    col_map = (col_map.detach().cpu().numpy() * 255).astype('uint8')
    imageio.imwrite(f'{savePath}/envmaps/{prtx}view_map.png', col_map)
    imageio.imwrite(f'{savePath}/envmaps/{prtx}ref_map.png', env_map)
    print(f"Using {render_mode} render mode")
    for idx, rays, gt_rgb in iterator():

        # rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
        #                                 ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        # rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map, debug_map, normal_map, env_map, col_map = brender(rays, tensorf, N_samples=N_samples,
                                     ndc_ray=ndc_ray, white_bg = white_bg)
        
        normal_map = (normal_map * 127 + 128).clamp(0, 255).byte()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if gt_rgb is not None:
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            # fig, axs = plt.subplots(2, 2)
            # axs[0, 0].imshow(rgb_map)
            # axs[1, 0].imshow(gt_rgb)
            # axs[0, 1].imshow(rgb_map-gt_rgb)
            # axs[1, 1].imshow(normal_map)
            # plt.show()

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                # ic(PSNRs[-1], ssim)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}normal_{idx:03d}.png', normal_map)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}debug_{idx:03d}.png', debug_map.numpy().astype(np.uint8))
            imageio.imwrite(f'{savePath}/envmaps/{prtx}ref_map_{idx:03d}.png', env_map)
            imageio.imwrite(f'{savePath}/envmaps/{prtx}view_map_{idx:03d}.png', col_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)
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

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

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
                yield idx, rays, gt_rgb
            else:
                yield idx, rays, None
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
            yield idx, rays, None
    return evaluate(iterator, test_dataset, tensorf, renderer, *args, ndc_ray=ndc_ray, **kwargs)
