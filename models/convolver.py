
import torch
import torch.nn.functional as F
from icecream import ic

def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

def combine_kernels1d(kernel1, kernel2):
    if kernel2 is None:
        return kernel1
    if kernel1 is None:
        return kernel2
    
    s1 = kernel1.shape[-1]
    s2 = kernel2.shape[-1]
    sf = s1 + s2 - 1
    p = (sf - s1) // 2 + 1
    kernel1 = kernel1.reshape(1, 1, s1)
    kernel2 = kernel2.reshape(1, 1, s2)
    kernelf = -F.conv1d(kernel1, kernel2, stride=1, padding=p)
    # place kernel1 at center
    return kernelf

def combine_kernels2d(kernel1, kernel2):
    if kernel2 is None:
        return kernel1
    if kernel1 is None:
        return kernel2
    
    s1 = kernel1.shape[-1]
    s2 = kernel2.shape[-1]
    sf = s1 + s2 - 1
    p = (sf - s1) // 2 + 1
    kernel1 = kernel1.reshape(1, 1, s1, s1)
    kernel2 = kernel2.reshape(1, 1, s2, s2)
    # place kernel1 at center
    kernelf = -F.conv2d(kernel1, kernel2, stride=1, padding=p)
    return kernelf

class Convolver(torch.nn.Module):
    def __init__(self, sizes, multi_scale, dtype=torch.float32):
        super().__init__()

        self.sizes = sizes
        self.dtype = dtype
        self.multi_scale = multi_scale
        f_blur = torch.tensor([0, 1, 0])
        f_edge = torch.tensor([-1, 0, 1]) / 2
        l = len(f_blur)

        self.sizes = [1, *self.sizes]

        self.register_buffer('dy_filter', (f_blur[None, :] * f_edge[:, None]).reshape(1, 1, l, l))
        self.register_buffer('dx_filter', self.dy_filter.permute(0, 1, 3, 2))
        self.register_buffer('dz_filter', f_edge.reshape(1, 1, l))

        self.interp_mode = 'bilinear'
        # self.interp_mode = 'bicubic'
        
    def get_kernels(self, size_weights=None, with_normals=False):
        if self.multi_scale and size_weights is not None and size_weights[1:].sum() > 0:
            plane_kerns = self.norm_plane_kernels
            line_kerns = self.norm_line_kernels
        else:
            plane_kerns = [outputs[:1] for outputs in self.norm_plane_kernels]
            line_kerns = [outputs[:1] for outputs in self.norm_line_kernels]
        if not with_normals:
            plane_kerns = plane_kerns[0:1]
            line_kerns = line_kerns[0:1]
        return plane_kerns, line_kerns

    def compute_size_weights(self, xyz_sampled, units):
        size = xyz_sampled[..., 3]#*3
        # the sum across the weights is 1
        # just want the closest grid point
        # first calculate the size of voxels in meters
        # voxel sizes go from smallest to largest
        voxel_sizes = self.sizes
        voxel_sizes = torch.tensor(voxel_sizes, dtype=torch.float32, device=xyz_sampled.device)*units.min()
        size_gap = voxel_sizes[:-1] - voxel_sizes[1:]

        # linear interpolation
        size_diff = size.reshape(1, -1) - voxel_sizes.reshape(-1, 1)
        inds = (size_diff > 0).int().sum(dim=0) - 1 # index of the largest element that size is greater than
        do_interp = (inds >= 0) & (inds < len(voxel_sizes)-1)
        weights = torch.zeros((len(voxel_sizes), *size.shape), dtype=torch.float32, device=size.device)
        # fill in values where the size is outside the range of sizes
        weights[0, inds<0] = 1
        weights[-1, inds==len(voxel_sizes)-1] = 1
        # inds+1
        # fill in interpolated values
        # ic(size_diff[inds[do_interp]+1, do_interp].shape, size_gap[inds[do_interp]].shape)
        # ic((size_diff[inds[do_interp]+1, do_interp] / size_gap[inds[do_interp]]).shape)
        weights[inds[do_interp]+1, do_interp] = -size_diff[inds[do_interp], do_interp] / size_gap[inds[do_interp]]
        weights[inds[do_interp], do_interp] = size_diff[inds[do_interp]+1, do_interp] / size_gap[inds[do_interp]]

        # voxel_sizes = [level.units.max() for level in self.levels]

        # then, the weight should be summed from the smallest supported size to the largest
        # size_diff = abs(voxel_sizes.reshape(1, -1) - size.reshape(-1, 1))/voxel_sizes.max()
        # weights = F.softmax(-size_diff.reshape(*size.shape, len(voxel_sizes)), dim=-1)
        return weights

    def set_smoothing(self, sm):
        print(f"Setting smoothing to {sm}")
        device = self.dx_filter.device
        self.norm_line_kernels = [[combine_kernels1d(gaussian_fn(2*s+3, std=s*sm).to(device), conv) for s in self.sizes] for conv in [None, self.dz_filter]]
        self.norm_plane_kernels = [[combine_kernels2d(gkern(2*s+3, std=s*sm).to(device), conv) for s in self.sizes] for conv in [None, self.dx_filter, self.dy_filter]]
        # self.norm_line_kernels = [[conv for s in self.sizes] for conv in [None, self.dz_filter]]
        # self.norm_plane_kernels = [[conv for s in self.sizes] for conv in [None, self.dx_filter, self.dy_filter]]

    def multi_size_plane(self, plane, coordinate_plane, size_weights, convs):
        # plane.shape: 1, n_comp, grid_size, grid_size
        # coordinate_plane.shape: 1, N, 1, 2
        # convs: [nested list of tensors of shape: 1, 1, s, s]. Outer list is for different outputs. Inner list is for different sizes.
        # returns: n_comp, N
        
        n_comp = plane.shape[1]
        num_scales = len(convs[0])
        num_outputs = len(convs)
        p_plane = plane.permute(1, 0, 2, 3)
        size_weights = size_weights.reshape(num_scales, 1, 1, -1) if num_scales > 1 else 1
        out = []

        for sizes in convs:
            for comb_kernel in sizes:
                if comb_kernel is None:
                    level = plane
                else:
                    s = comb_kernel.shape[-1]
                    level = F.conv2d(p_plane, comb_kernel.reshape(1, -1, s, s), stride=1, padding=s//2)
                    level = level.permute(1, 0, 2, 3)
                out.append(level)

        out = torch.cat(out, dim=1)
        
        ms_plane_coef = F.grid_sample(out, coordinate_plane, mode=self.interp_mode, align_corners=True)
        # ms_line_coef.shape: len(self.line_kernels), len(convs), n_comp, N
        ms_plane_coef = ms_plane_coef.reshape(num_outputs, num_scales, n_comp, -1).permute(1, 0, 2, 3)
        # line_coef.shape: len(convs), n_comp, N
        plane_coef = (ms_plane_coef * size_weights).sum(dim=0)

        output = [plane_coef[i] for i in range(num_outputs)]
        if len(output) == 1:
            return output[0]
        return output

    def multi_size_line(self, line, coordinate_line, size_weights, convs):
        # plane.shape: 1, n_comp, grid_size, 1
        # coordinate_plane.shape: 1, N, 1, 2
        # convs: [nested list of tensors of shape: 1, 1, s]. Outer list is for different outputs. Inner list is for different sizes.
        # returns: n_comp, N
        
        n_comp = line.shape[1]
        num_scales = len(convs[0])
        num_outputs = len(convs)
        size_weights = size_weights.reshape(num_scales, 1, 1, -1) if num_scales > 1 else 1
        p_line = line.permute(1, 0, 2, 3).squeeze(-1)
        out = []
        for kernels in convs:
            for comb_kernel in kernels:
                if comb_kernel is None:
                    level = line
                else:
                    s = comb_kernel.shape[-1]
                    level = F.conv1d(p_line, comb_kernel.reshape(1, 1, s), stride=1, padding=s//2)
                    level = level.unsqueeze(-1).permute(1, 0, 2, 3)
                out.append(level)
        out = torch.cat(out, dim=1)
        
        ms_line_coef = F.grid_sample(out, coordinate_line, mode=self.interp_mode, align_corners=True)
        ms_line_coef = ms_line_coef.reshape(num_outputs, num_scales, n_comp, -1).permute(1, 0, 2, 3)
        # line_coef.shape: len(convs), n_comp, N
        line_coef = (ms_line_coef * size_weights).sum(dim=0)
        # deliberately split the result to avoid confusion
        output = [line_coef[i] for i in range(num_outputs)]
        if num_outputs == 1:
            return output[0]
        return output