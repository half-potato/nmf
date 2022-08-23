import torch
import torch.nn.functional as F
from icecream import ic

SIGN = -1
# sign = -1 matches grid sample

def gaussian_fn(M, std, **kwargs):
    n = torch.arange(0, M, **kwargs) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128, **kwargs):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std, **kwargs) 
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

def combine_kernels3d(kernel1, kernel2):
    if kernel2 is None:
        return kernel1
    if kernel1 is None:
        return kernel2
    
    s1 = kernel1.shape[-1]
    s2 = kernel2.shape[-1]
    sf = s1 + s2 - 1
    p = (sf - s1) // 2 + 1
    kernel1 = kernel1.reshape(1, 1, s1, s1, s1)
    kernel2 = kernel2.reshape(1, 1, s2, s2, s2)
    # place kernel1 at center
    kernelf = -F.conv3d(kernel1, kernel2, stride=1, padding=p)
    return kernelf

class GridSampler2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='zeros', align_corners=None, smoothing=0):
        # plane.shape: 1, n_comp, grid_size, grid_size
        # grid: (1, N, 1, 2)
        ctx.save_for_backward(input, grid)
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.smoothing = smoothing
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_grid = None
        is_3d = len(grid.shape) == 5
        if ctx.needs_input_grad[1]:
            f_blur = torch.tensor([0.5, 0.5], device=grid.device)
            f_edge = SIGN*torch.tensor([1, -1], device=grid.device) / 2
            l = len(f_blur)
            kernlen = 2*int(ctx.smoothing)+1
            if is_3d:
                dy_filter = (f_blur[None, :, None] * f_edge[:, None, None] * f_blur[None, None, :]).reshape(1, 1, l, l, l)
                dx_filter = dy_filter.permute(0, 1, 3, 2, 4)
                dz_filter = dy_filter.permute(0, 1, 2, 4, 3)

                g1 = gaussian_fn(kernlen, std=ctx.smoothing+1e-8, device=grad_output.device)
                smooth_kern = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
                smooth_kern /= smooth_kern.sum()
                sm_dx_filter = combine_kernels3d(smooth_kern, dx_filter)
                sm_dy_filter = combine_kernels3d(smooth_kern, dy_filter)
                sm_dz_filter = combine_kernels3d(smooth_kern, dz_filter)
                s = sm_dx_filter.shape[-1]

                pinput = input.permute(1, 0, 2, 3, 4)
                dx_input = F.conv3d(pinput, sm_dx_filter.reshape(1, 1, s, s, s), stride=1, padding=s//2)
                dy_input = F.conv3d(pinput, sm_dy_filter.reshape(1, 1, s, s, s), stride=1, padding=s//2)
                dz_input = F.conv3d(pinput, sm_dz_filter.reshape(1, 1, s, s, s), stride=1, padding=s//2)

                dx = F.grid_sample(dx_input.permute(1, 0, 2, 3, 4), grid, mode=ctx.mode, padding_mode=ctx.padding_mode, align_corners=ctx.align_corners)
                dy = F.grid_sample(dy_input.permute(1, 0, 2, 3, 4), grid, mode=ctx.mode, padding_mode=ctx.padding_mode, align_corners=ctx.align_corners)
                dz = F.grid_sample(dy_input.permute(1, 0, 2, 3, 4), grid, mode=ctx.mode, padding_mode=ctx.padding_mode, align_corners=ctx.align_corners)

                grad_grid = torch.stack([(grad_output*dx).sum(dim=1), (grad_output*dy).sum(dim=1), (grad_output*dz).sum(dim=1)], dim=-1)
            else:
                dy_filter = (f_blur[None, :] * f_edge[:, None]).reshape(1, 1, l, l)
                dx_filter = dy_filter.permute(0, 1, 3, 2)

                smooth_kern = gkern(2*int(ctx.smoothing)+1, std=ctx.smoothing+1e-8, device=grad_output.device)
                smooth_kern /= smooth_kern.sum()
                sm_dx_filter = combine_kernels2d(smooth_kern, dx_filter)
                sm_dy_filter = combine_kernels2d(smooth_kern, dy_filter)
                s = sm_dx_filter.shape[-1]

                dx_input = F.conv2d(input.permute(1, 0, 2, 3), sm_dx_filter.reshape(1, -1, s, s), stride=1, padding=s//2)
                dy_input = F.conv2d(input.permute(1, 0, 2, 3), sm_dy_filter.reshape(1, -1, s, s), stride=1, padding=s//2)

                dx = F.grid_sample(dx_input.permute(1, 0, 2, 3), grid, mode=ctx.mode, padding_mode=ctx.padding_mode, align_corners=ctx.align_corners)
                dy = F.grid_sample(dy_input.permute(1, 0, 2, 3), grid, mode=ctx.mode, padding_mode=ctx.padding_mode, align_corners=ctx.align_corners)

                grad_grid = torch.stack([(grad_output*dx).sum(dim=1), (grad_output*dy).sum(dim=1)], dim=-1)

        grad_input = None
        if ctx.needs_input_grad[0]:
        #  if True:
            if ctx.mode == "bilinear":
                mode_enum = 0
            elif ctx.mode == "nearest":
                mode_enum = 1
            else:  # mode == 'bicubic'
                mode_enum = 2

            if ctx.padding_mode == "zeros":
                padding_mode_enum = 0
            elif ctx.padding_mode == "border":
                padding_mode_enum = 1
            else:  # padding_mode == 'reflection'
                padding_mode_enum = 2
            if is_3d:
                op = torch._C._jit_get_operation('aten::grid_sampler_3d_backward')
                if type(op) == tuple:
                    op = op[0]
                grad_input, _ = op(grad_output, input, grid, mode_enum, padding_mode_enum, ctx.align_corners)
            else:
                op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
                if type(op) == tuple:
                    op = op[0]
                grad_input, _ = op(grad_output, input, grid, mode_enum, padding_mode_enum, ctx.align_corners, (ctx.needs_input_grad[0], False))

        return grad_input, grad_grid, None, None, None, None

class GridSampler1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='zeros', align_corners=None, smoothing=0):
        # plane.shape: 1, n_comp, grid_size, 1
        # grid: (1, N, 1, 2)
        ctx.save_for_backward(input, grid)
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.smoothing = smoothing
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        smoothing = ctx.smoothing
        grad_grid = None
        if ctx.needs_input_grad[1]:
            # f_edge = torch.tensor([-1, 0, 1]) / 2
            # f_edge = torch.tensor([1, 0, -1]) / 2
            f_edge = SIGN*torch.tensor([1, -1]) / 2
            l = len(f_edge)

            dz_filter = f_edge.reshape(1, 1, l)
            smooth_kern = gaussian_fn(2*int(smoothing)+3, std=smoothing, device=grad_output.device)
            sm_dx_filter = combine_kernels2d(smooth_kern, dz_filter)
            s = sm_dx_filter.shape[-1]

            dx_input = F.conv1d(input, sm_dx_filter.reshape(1, 1, s), stride=1, padding=s//2)

            grad_grid = grad_output * F.grid_sample(dx_input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

        grad_input = None
        if ctx.needs_input_grad[0]:
            if mode == "bilinear":
                mode_enum = 0
            elif mode == "nearest":
                mode_enum = 1
            else:  # mode == 'bicubic'
                mode_enum = 2

            if padding_mode == "zeros":
                padding_mode_enum = 0
            elif padding_mode == "border":
                padding_mode_enum = 1
            else:  # padding_mode == 'reflection'
                padding_mode_enum = 2
            op = torch._C._jit_get_operation('aten::grid_sampler_1d_backward')
            

            grad_input, _ = op(grad_output, input, grid, mode_enum, padding_mode_enum, align_corners, (ctx.needs_input_grad[1], False))
        return grad_input, grad_grid, None, None, None, None
    
def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None, smoothing=0):
    # plane.shape: 1, n_comp, grid_size, 1
    # grid: (1, N, 1, 2)
    if grid.shape[-1] == 2 or grid.shape[-1] == 3:
        return GridSampler2D.apply(input, grid, mode, padding_mode, align_corners, smoothing)
    else:
        raise NotImplementedError("GridSampler only implemented for 2D/3D inputs")
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    im = cv2.imread('plane.png')
    s = im.shape[1]
    plt.imshow(im)
    plt.figure()
    im = torch.as_tensor(im, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    x, y = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing='ij')
    grid = torch.stack([x, y], dim=-1).unsqueeze(0).reshape(1, -1, 1, 2)
    ic(grid.shape)
    grid.requires_grad = True
    dist = torch.linalg.norm(grid, dim=-1, keepdim=True) + 1e-8
    ic(dist.shape)
    grid = torch.where(dist > 0.5, (1 - 1/dist), dist) * grid/dist
    ic(grid)
    plt.imshow(grid.detach().numpy()[0, :, :, 0].reshape(s, s))
    plt.figure()
    plt.imshow(im[0].permute(1, 2, 0).detach().numpy())
    samp_im = grid_sample(im, grid, smoothing=0)
    ic(samp_im.shape)
    grad_outputs = torch.ones(samp_im.shape)
    plt.imshow(samp_im.detach().reshape(3, s, s).permute(1, 2, 0).numpy())
    plt.figure()
    g = torch.autograd.grad(samp_im, grid, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    ic(grid.shape)
    ic(g.shape)
    plt.imshow(g[..., 0].detach().reshape(s, s))
    plt.figure()
    plt.imshow(g[..., 1].detach().reshape(s, s))
    plt.show()
