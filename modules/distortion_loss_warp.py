import warp as wp
from .distortion_loss_pseudo import distortion_loss_pseudo
from icecream import ic
import torch

wp.init()
@wp.kernel
def distortion_bidir_kernel(
    midpoint: wp.array2d(dtype=float),
    full_weight: wp.array2d(dtype=float),
    dt: wp.array2d(dtype=float),
    dm: wp.array2d(dtype=float),
    dw: wp.array2d(dtype=float),
    ddt: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
    M: int):
    b, i, j = wp.tid()
    mp1 = midpoint[b, i]
    fw1 = full_weight[b, i]
    pdt = dt[b, i]

    inner = fw1 * fw1 * pdt / 3.0
    # backward
    ddt[b, i] = fw1*fw1
    dw[b, i] =  2.0*fw1*pdt / 3.0

    inter = float(0.0)
    dw1 = float(0.0)
    dm1 = float(0.0)
    for j in range(M):
        mp2 = midpoint[b, j]
        fw2 = full_weight[b, j]

        aut = mp1 - mp2
        wm = fw1 * fw2
        dut = wp.abs(aut)
        inter += dut * wm
        s = torch.sign(aut)

        wp.atomic_add(dw, b, j, dut*fw1)
        wp.atomic_add(dm, b, j, -wm * s)
        dw1 += dut*fw2
        dm1 += wm * s
    dw[b, i] = dw1
    dm[b, i] = dm1

    wp.atomic_add(loss, 0, inter+inner)
    wp.atomic_add(loss, 0, inter)


def distortion_bidir(midpoint, full_weight, dt):
    B, M = midpoint.shape
    device = midpoint.device
    device = 'cuda'
    dtype = midpoint.dtype
    dtype = float

    midpoint_wp = wp.from_torch(midpoint)
    full_weight_wp = wp.from_torch(full_weight)
    dt_wp = wp.from_torch(dt)
    loss = wp.array([0.0], dtype=dtype, device=device)
    dm = wp.zeros((B, M), dtype=dtype, device=device)
    dw = wp.zeros((B, M), dtype=dtype, device=device)
    ddt = wp.zeros((B, M), dtype=dtype, device=device)

    wp.launch(kernel=distortion_bidir_kernel,
              dim=(B, M),
              inputs=[midpoint_wp, full_weight_wp, dt_wp, dm, dw, ddt, loss, M],
              device=device)
    n = B
    loss = wp.to_torch(loss)/n
    dm = wp.to_torch(dm)/n
    dw = wp.to_torch(dw)/n
    ddt = wp.to_torch(ddt)/n

    return loss, dm, dw, ddt


class _DistortionLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, midpoint, full_weight, dt):
        accum, dm, dw, dt = distortion_bidir(midpoint, full_weight, dt)
        # ic(full_weight.mean())

        # ic(dm, dw, dt)
        ctx.save_for_backward(dm, dw, dt)
        return accum

    @staticmethod
    def backward(ctx, daccum):
        dm, dw, dt = ctx.saved_tensors
        return daccum * dm, daccum * dw, daccum * dt

distortion_loss = _DistortionLoss.apply

if __name__ == "__main__":
    B = 3
    M = 1000
    device = torch.device('cuda')
    dtype = torch.float
    midpoint = torch.rand(B, M, dtype=dtype, device=device)
    full_weight = torch.rand(B, M, dtype=dtype, device=device)
    dt = torch.rand(B, M, dtype=dtype, device=device)
    midpoint.requires_grad = True 
    full_weight.requires_grad = True 
    dt.requires_grad = True 
    print(distortion_loss_pseudo(midpoint, full_weight, dt))
    print(distortion_loss(midpoint, full_weight, dt))
    # torch.autograd.gradcheck(distortion_loss_pseudo, (midpoint, full_weight, dt))
    # torch.autograd.gradcheck(distortion_loss, (midpoint, full_weight, dt))
