import torch
import math
from icecream import ic

class ERBrightSampler():
    def __init__(self):
        pass

    def sample(self, bg_module, N, eps=torch.finfo(torch.float32).eps):
        brightness = bg_module.activation_fn(bg_module.bg_mat[0]).mean(dim=-1)
        # multiply by sin of elevation
        device = brightness.device
        H, W = brightness.shape
        sin_vals = torch.sin(torch.arange(H, device=device) / H * math.pi)
        prob = brightness * sin_vals.reshape(-1, 1).expand(H, W)
        brightsum = prob.sum()
        prob = prob / brightsum
        cdf = torch.cumsum(prob.reshape(-1), dim=0)
        # plug random values into inverse cdf
        indices = ((torch.rand((N, 1), device=device) < cdf.reshape(1, -1)) * torch.arange(cdf.shape[0], device=device).reshape(1, -1)).max(dim=-1).indices
        # indices to angles
        row = indices // W
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
