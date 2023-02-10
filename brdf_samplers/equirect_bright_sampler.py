import torch
import math
from icecream import ic
import warp as wp
from modules.distortion_loss_warp import from_torch
import time

wp.init()

@wp.kernel
def kern_inv_cdf(
        cdf: wp.array(dtype=float),
        rand_vals: wp.array(dtype=float),
        output: wp.array(dtype=wp.int32),
        cdf_len: int,
        samp_len: int):

    ti = wp.tid()
    if ti < samp_len:
        v = rand_vals[ti]
        found = int(0) # no booleans?
        for i in range(cdf_len):
            if cdf[i] > v and found == 0:
                output[ti] = i
                found = 1
        if found == 0:
            output[ti] = cdf_len-1
                # break
                # break is not supported

def inv_cdf_wp(cdf, rand_vals):
    device = cdf.device
    cdf_len = cdf.shape[0]
    samp_len = rand_vals.shape[0]
    output = -torch.ones((samp_len), device=device, dtype=torch.int32)

    wcdf = from_torch(cdf, requires_grad=False)
    wrand_vals = from_torch(rand_vals, requires_grad=False)
    woutput = from_torch(output, dtype=wp.int32, requires_grad=False)
    wp.launch(kernel=kern_inv_cdf,
              dim=(samp_len),
              inputs=[wcdf, wrand_vals, woutput, cdf_len, samp_len],
              device='cuda')
    return output

def inv_cdf(cdf, rand_vals):
    device = cdf.device
    indices = ((rand_vals.reshape(-1, 1) < cdf.reshape(1, -1)) * torch.arange(cdf.shape[0], device=device).reshape(1, -1)).max(dim=-1).indices
    return indices

class ERBrightSampler():
    def __init__(self):
        pass

    def sample(self, bg_module, N, eps=torch.finfo(torch.float32).eps):
        brightness = bg_module.get_brightness()
        # multiply by sin of elevation
        device = brightness.device
        H, W = brightness.shape
        sin_vals = torch.sin(torch.arange(H, device=device) / H * math.pi)
        prob = brightness * sin_vals.reshape(-1, 1).expand(H, W)
        brightsum = brightness.sum()
        prob = brightness / brightsum
        cdf = torch.cumsum(prob.reshape(-1), dim=0).reshape(-1)
        rand_vals = torch.rand((N), device=device)
        indices = inv_cdf_wp(cdf, rand_vals).long()
        # indices = inv_cdf(cdf, rand_vals)
        # indices2 = inv_cdf(cdf, rand_vals)
        # ic(indices, indices2, rand_vals, cdf[indices], cdf[indices2])

        # plug random values into inverse cdf
        # ic(cdf.shape, N)
        # indices to angles
        row = torch.div(indices, W, rounding_mode='floor')
        col = indices % W

        # here is the angles to indices function
        #
        # a, b, c = viewdirs[:, 0:1], viewdirs[:, 1:2], viewdirs[:, 2:3]
        # norm2d = torch.sqrt(a**2+b**2)
        # phi = safemath.atan2(b, a)
        # theta = safemath.atan2(c, norm2d)
        # coords = torch.cat([
        #     (phi % (2*math.pi) - math.pi) / math.pi,
        #     -theta/math.pi*2,
        # ], dim=1)
        # x = coords.reshape(1, 1, -1, 2)
        x = row / H * 2 - 1
        y = col / W * 2 - 1
        phi = x * math.pi + math.pi
        theta = -y * math.pi / 2
        viewdirs = torch.stack([
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta),
        ], dim=-1).reshape(-1, 3)

        probs = prob[row, col] / (2 * math.pi * math.pi * torch.sin(theta + math.pi/2)).clip(min=eps)
        return viewdirs, probs, brightsum
