import cv2,torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.signal
import torch.nn as nn
from itertools import product 
from icecream import ic

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)




__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x):
        if x.shape[-1] == 1:
            h_tv = x[:,:,1:,:]-x[:,:,:-1,:]
            return h_tv.abs().mean()
        else:
            h_tv = x[:,:,1:,:-1]-x[:,:,:-1,:-1]
            w_tv = x[:,:,:-1,1:]-x[:,:,:-1,:-1]
            
            return (w_tv**2 + h_tv**2 + 1e-5).sqrt().mean()


    def _tensor_size(self,t):
        return t[0].numel()

import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def snells_law(r, n, l):
    # n: (B, 3) surface outward normal
    # l: (B, 3) light direction towards surface
    # r: ratio between indices of refraction. n1/n2
    # where n1 = index where light starts and n2 = index after surface penetration
    dtype = n.dtype
    n = n.double()
    l = l.double()

    cosi = torch.matmul(n.reshape(-1, 1, 3), l.reshape(-1, 3, 1)).reshape(*n.shape[:-1], 1)
    Nsign = torch.sign(cosi)
    N = torch.where(cosi < 0, n, -n)
    cosi = cosi * Nsign
    R = torch.where(cosi < 0, 1/r, r)

    k = 1 - R * R * (1 - cosi * cosi);
    refractdir = R * l + (R * cosi - torch.sqrt(k.clip(min=0))) * N

    # c = -torch.matmul(n.reshape(-1, 1, 3), l.reshape(-1, 3, 1)).reshape(*n.shape[:-1], 1)
    # sign = torch.sign(c).abs()
    # refractdir = (r*l + (r * c.abs() - torch.sqrt( (1 - r**2 * (1-c**2)).clip(min=1e-8) )) * sign*n)
    return refractdir.type(dtype)

def fresnel_law(ior1, ior2, n, l, o):
    # input: 
    #  n: (B, 3) surface outward normal
    #  l: (B, 3) light direction towards surface
    #  o: (B, 3) refracted light direction given by snells_law
    #  ior1: index of refraction for material from which light was emitted
    #  ior2: index of refraction for material after surface
    # output:
    #  ratio reflected, between 0 and 1
    cos_i = torch.matmul(n.reshape(-1, 1, 3), l.reshape(-1, 3, 1)).reshape(*n.shape[:-1], 1)
    cos_t = torch.matmul(n.reshape(-1, 1, 3), o.reshape(-1, 3, 1)).reshape(*n.shape[:-1], 1)
    sin_t = torch.sqrt(1 - cos_t**2)
    s_polar = (ior2 * cos_i - ior1 * cos_t) / (ior2 * cos_i + ior1 * cos_t)
    p_polar = (ior2 * cos_t - ior1 * cos_i) / (ior2 * cos_t + ior1 * cos_i)
    ratio_reflected = (s_polar + p_polar)/2
    return torch.where(sin_t >= 1, torch.ones_like(ratio_reflected), ratio_reflected)

def refract_reflect(ior1, ior2, n, l, p):
    # n: (B, 3) surface outward normal
    # l: (B, 3) light direction towards surface
    # p: (B) reflectivity of material, between 0 and 1
    # ior1: index of refraction for material from which light was emitted
    # ior2: index of refraction for material after surface
    ratio = ior2/ior1
    o = snells_law(ratio, n, l)
    ratio_reflected = fresnel_law(ior1, ior2, n, l, o)
    ratio_refracted = 1 - ratio_reflected
    out_ratio_reflected = 1 - p * ratio_refracted
    return out_ratio_reflected

class AlphaGridMask(torch.nn.Module):
    def __init__(self, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.register_buffer('aabb', aabb)

        aabbSize = self.aabb[1] - self.aabb[0]
        invgrid_size = 1.0/aabbSize * 2
        grid_size = torch.LongTensor(
            [alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]])
        self.register_buffer('grid_size', grid_size)
        self.register_buffer('invgrid_size', invgrid_size)
        self.register_buffer('alpha_volume', alpha_volume)

    def sample_alpha(self, xyz_sampled, contract_space=False):
        xyz_sampled = self.normalize_coord(xyz_sampled, contract_space)
        H, W, D = self.alpha_volume.shape
        i = ((xyz_sampled[..., 0]/2+0.5)*(H-1)).long()
        j = ((xyz_sampled[..., 1]/2+0.5)*(W-1)).long()
        k = ((xyz_sampled[..., 2]/2+0.5)*(D-1)).long()
        alpha_vals = self.alpha_volume[i, j, k]
        # alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled[..., :3].view(
        #     1, -1, 1, 1, 3), align_corners=False).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled, contract_space):
        coords = (xyz_sampled[..., :3]-self.aabb[0]) * self.invgrid_size - 1
        size = xyz_sampled[..., 3:4]
        normed = torch.cat((coords, size), dim=-1)
        if contract_space:
            dist = torch.linalg.norm(normed[..., :3], dim=-1, keepdim=True, ord=torch.inf) + 1e-8
            direction = normed[..., :3] / dist
            contracted = torch.where(dist > 1, (2-1/dist), dist)/2 * direction
            return torch.cat([ contracted, xyz_sampled[..., 3:] ], dim=-1)
        else:
            return normed

    def contract_coord(self, xyz_sampled): 
        dist = torch.linalg.norm(xyz_sampled[..., :3], dim=1, keepdim=True) + 1e-8
        direction = xyz_sampled[..., :3] / dist
        contracted = torch.where(dist > 1, (2-1/dist), dist) * direction
        return torch.cat([ contracted, xyz_sampled[..., 3:] ], dim=-1)

def log_lerp(t, v0, v1):
  """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
  if v0 <= 0 or v1 <= 0:
    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
  lv0 = np.log(v0)
  lv1 = np.log(v1)
  return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.

  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.

  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.

  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  return delay_rate * log_lerp(step / max_steps, lr_init, lr_final)
