import warp as wp
from .distortion_loss_pseudo import distortion_loss_pseudo, lossfun_distortion
from icecream import ic
import torch
import time

# wrap a torch tensor to a wp array, data is not copied
def from_torch(t, dtype=None, requires_grad=None):
    # ensure tensors are contiguous
    assert(t.is_contiguous())
    if (t.dtype != torch.float32 and t.dtype != torch.int32):
        raise RuntimeError("Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type")
    if dtype is None:
        dtype = wp.types.float32 if t.dtype == torch.float32 else wp.types.int32

    requires_grad = requires_grad if requires_grad is not None else t.requires_grad
    # if target is a vector or matrix type
    # then check if trailing dimensions match
    # the target type and update the shape
    if hasattr(dtype, "_shape_"):
        
        try:
            num_dims = len(dtype._shape_)
            type_dims = dtype._shape_
            source_dims = t.shape[-num_dims:]

            for i in range(len(type_dims)):
                if source_dims[i] != type_dims[i]:
                    raise RuntimeError()

            shape = t.shape[:-num_dims]

        except:
            raise RuntimeError(f"Could not convert source Torch tensor with shape {t.shape}, to Warp array with dtype={dtype}, ensure that trailing dimensions match ({source_dims} != {type_dims}")
    
    else:
        shape = t.shape

    a = wp.types.array(
        ptr=t.data_ptr(),
        dtype=dtype,
        shape=shape,
        copy=False,
        owner=False,
        requires_grad=requires_grad,
        device=t.device.type)

    # save a reference to the source tensor, otherwise it will be deallocated
    a.tensor = t
    return a

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

        # wp.atomic_add(dw, b, j, dut*fw1)
        # wp.atomic_add(dm, b, j, -wm * s)
        dw1 += dut*fw2
        dm1 += wm * s
    dw[b, i] = dw1*2.0 + 2.0*fw1*pdt / 3.0
    dm[b, i] = dm1*2.0

    wp.atomic_add(loss, 0, inter+inner)


def distortion_bidir(midpoint, full_weight, dt):
    B, M = midpoint.shape
    device = 'cuda'
    dtype = float


    midpoint_wp = from_torch(midpoint, requires_grad=False)
    full_weight_wp = from_torch(full_weight, requires_grad=False)
    dt_wp = from_torch(dt, requires_grad=False)
    # loss = wp.array([0.0], dtype=dtype, device=device)
    # dm = wp.zeros((B, M), dtype=dtype, device=device)
    # dw = wp.zeros((B, M), dtype=dtype, device=device)
    # ddt = wp.zeros((B, M), dtype=dtype, device=device)
    device = midpoint.device
    dtype = midpoint.dtype
    loss = torch.zeros((1), dtype=dtype, device=device)
    dm = torch.zeros((B, M), dtype=dtype, device=device)
    dw = torch.zeros((B, M), dtype=dtype, device=device)
    ddt = torch.zeros((B, M), dtype=dtype, device=device)
    loss_wp = from_torch(loss, requires_grad=False)
    dm_wp = from_torch(dm, requires_grad=False)
    dw_wp = from_torch(dw, requires_grad=False)
    ddt_wp = from_torch(ddt, requires_grad=False)

    wp.launch(kernel=distortion_bidir_kernel,
              dim=(B, M),
              inputs=[midpoint_wp, full_weight_wp, dt_wp, dm_wp, dw_wp, ddt_wp, loss_wp, M],
              device='cuda')
    n = B
    loss = loss/n
    dm = dm/n
    dw = dw/n
    ddt = ddt/n

    return loss, dm, dw, ddt


class _DistortionLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, midpoint, full_weight, dt):
        accum, dm, dw, dt = distortion_bidir(midpoint, full_weight, dt)
        # ic(full_weight.mean())

        # ic(dm, dw, dt)
        ctx.save_for_backward(dm, dw, dt)
        return accum[0]

    @staticmethod
    def backward(ctx, daccum):
        dm, dw, dt = ctx.saved_tensors
        return daccum * dm, daccum * dw, daccum * dt

calc_distortion_loss = _DistortionLoss.apply

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
    gt_loss = lossfun_distortion(midpoint, full_weight, dt)
    w_loss = distortion_loss(midpoint, full_weight, dt)
    ic(gt_loss, w_loss)

    deriv = torch.ones_like(w_loss)
    dmat1 = torch.autograd.grad(gt_loss, midpoint, deriv, allow_unused=True, retain_graph=True)
    dmat2 = torch.autograd.grad(w_loss, midpoint, deriv, allow_unused=True, retain_graph=True)
    ic(dmat1, dmat2)
    dmat1 = torch.autograd.grad(gt_loss, full_weight, deriv, allow_unused=True)
    dmat2 = torch.autograd.grad(w_loss, full_weight, deriv, allow_unused=True)
    ic(dmat1, dmat2)
    # torch.autograd.gradcheck(distortion_loss_pseudo, (midpoint, full_weight, dt))
    torch.autograd.gradcheck(distortion_loss, (midpoint, full_weight, dt))
