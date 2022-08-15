import torch
from icecream import ic

import triton
import triton.language as tl

@triton.jit
def distortion_bidir_kernel(
        midpoint_ptr, full_weight_ptr, dt_ptr,
        accum_ptr, dm_ptr, dw_ptr, ddt_ptr,
        B: tl.constexpr, M: tl.constexpr,
        strideb: tl.constexpr, stridem: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr):
    # midpoint: (B, M)
    # full_weight: (B, M)
    # dt: (B, M)
    # dmp1 = tl.zeros((BLOCK_SIZE_M))
    # dmp2 = tl.zeros((BLOCK_SIZE_M))
    # dw1 = tl.zeros((BLOCK_SIZE_M))
    # dw2 = tl.zeros((BLOCK_SIZE_M))
    # ddt = tl.zeros((BLOCK_SIZE_M))

    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_n = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_a = pid_b * M * stridem +  offs_m * stridem
    offs_b = pid_b * M * stridem +  offs_n * stridem
    print(M, offs_a, offs_b, stridem)
    
    mask_a = (offs_a < ((pid_b+1) * M)) & (offs_m < M)
    mask_b = (offs_b < ((pid_b+1) * M)) & (offs_n < M)

    # tl.store(dm_ptr + offs_a, offs_a*0+pid_b, mask_a)

    w1 = tl.load(full_weight_ptr + offs_a, mask_a)
    w2 = tl.load(full_weight_ptr + offs_b, mask_b)
    mp1 = tl.load(midpoint_ptr + offs_a, mask_a)
    mp2 = tl.load(midpoint_ptr + offs_b, mask_b)
    pdt = tl.load(dt_ptr + offs_a, mask_a)
    print(w1, w2)
    inner_loss = tl.sum(tl.ravel(w1 * w1 * pdt), axis=0) / 3

    # forward
    dut = mp1[:, None] - mp2[None, :]
    wm = w1[:, None] * w1[None, :]
    # inter_loss = tl.sum(tl.ravel(tl.abs(dut) * wm), axis=0)
    inter_loss = tl.sum(tl.ravel(wm), axis=0)
    print(accum_ptr, inter_loss)
    tl.atomic_add(accum_ptr, inter_loss)
    """

    # backward

    dw2 = tl.sum((tl.abs(dut)*w1[:, None]), axis=0)
    dw1 = tl.sum((tl.abs(dut)*w2[None, :]), axis=1)

    # s = tl.sign(dut)
    s = tl.where(dut > 0, 1, -1)
    dmp2 = tl.sum(-(wm * s), axis=0)
    dmp1 = tl.sum((wm * s), axis=1)

    # forward
    inner_loss = tl.sum(tl.ravel(w1 * w1 * pdt), axis=0) / 3
    print(pdt, w1, tl.ravel(w1 * w1 * pdt), tl.sum(tl.ravel(w1 * w1 * pdt), axis=0) / 3)
    # backward
    ddt = w1 * w1
    dw1 = 2 * w1 * pdt

    # accumulate gradients
    # tl.atomic_add(dm_ptr + offs_a, dmp1[:, None], mask_a)
    # tl.atomic_add(dm_ptr + offs_a, offs_a, mask_a)
    # tl.atomic_add(dm_ptr + offs_b, dmp2[:, None], mask_b)
    tl.atomic_add(dw_ptr + offs_a, dw1[:, None], mask_a)
    tl.atomic_add(dw_ptr + offs_b, dw2[:, None], mask_b)
    tl.atomic_add(ddt_ptr + offs_a, ddt[:, None], mask_a)
    # tl.atomic_add(accum_ptr, inner_loss + inter_loss)
    inner_loss = tl.sum(tl.ravel(w1 * w1 * pdt), axis=0) / 3
    tl.atomic_add(accum_ptr, inner_loss)
    """

def distortion_bidir(midpoint, full_weight, dt):
    B, M = midpoint.shape
    BLOCK_SIZE_M = 16
    device = midpoint.device
    dtype = midpoint.dtype
    accum = torch.zeros((1), dtype=dtype, device=device)
    dm = torch.zeros((B, M), dtype=dtype, device=device)
    dw = torch.zeros((B, M), dtype=dtype, device=device)
    ddt = torch.zeros((B, M), dtype=dtype, device=device)

    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = (B, triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(M, BLOCK_SIZE_M))
    ic(midpoint.stride(0), midpoint.stride(1))
    ic(full_weight.stride(0), full_weight.stride(1))
    ic(dt.stride(0), dt.stride(1))
    ic(dm.stride(0), dm.stride(1))

    distortion_bidir_kernel[grid](
        midpoint.contiguous(), full_weight.contiguous(), dt.contiguous(),
        accum, dm, dw, ddt,
        B, M, dm.stride(0), dm.stride(1), BLOCK_SIZE_M)
    return accum, dm, dw, ddt

@torch.no_grad()
def distortion_bidir_pseudo(midpoint, full_weight, dt):
    # midpoint: (B, M)
    # full_weight: (B, M)
    # dt: (B, M)
    B, M = midpoint.shape
    BLOCK_SIZE_M = 128
    device = midpoint.device
    dm = torch.zeros((B, M), device=device)
    dw = torch.zeros((B, M), device=device)
    ddt = torch.zeros((B, M), device=device)
    accum = 0
    for b in range(0, B):
        for m in range(0, M, BLOCK_SIZE_M):
            w1 = full_weight[b, m : m+BLOCK_SIZE_M]
            mp1 = midpoint[b, m : m+BLOCK_SIZE_M]
            dw1 = dw[b, m : m+BLOCK_SIZE_M]
            dmp1 = dm[b, m : m+BLOCK_SIZE_M]
            for n in range(0, M, BLOCK_SIZE_M):
                mp2 = midpoint[b, n : n+BLOCK_SIZE_M]
                w2 = full_weight[b, n : n+BLOCK_SIZE_M]

                # forward
                dut = (mp1.reshape(-1, 1) - mp2.reshape(1, -1))
                wm = (w1.reshape(-1, 1) * w2.reshape(1, -1))
                # accum += (dut.abs()*wm).sum()

                # backward
                dw2 = dw[b, n : n+BLOCK_SIZE_M]
                dmp2 = dm[b, n : n+BLOCK_SIZE_M]

                dw2 += (dut.abs()*w1.reshape(-1, 1)).sum(dim=0)
                dw1 += (dut.abs()*w2.reshape(1, -1)).sum(dim=1)

                s = torch.sign(dut)
                dmp2 -= (wm * s).sum(dim=0)
                dmp1 += (wm * s).sum(dim=1)

            pdt = dt[b, m : m+BLOCK_SIZE_M ]

            # forward
            accum += (w1**2 * pdt).sum() / 3
            # backward
            ddt[b, m : m+BLOCK_SIZE_M ] += w1**2
            dw[b, m : m+BLOCK_SIZE_M] += 2 * w1 * pdt
    return accum, dm, dw, ddt

class _DistortionLossPseudo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, midpoint, full_weight, dt):
        accum, dm, dw, dt = distortion_bidir_pseudo(midpoint, full_weight, dt)
        ic(dm, dw, dt)

        ctx.save_for_backward(dm, dw, dt)
        return accum

    @staticmethod
    def backward(ctx, daccum):
        dm, dw, dt = ctx.saved_tensors
        return daccum * dm, daccum * dw, daccum * dt

distortion_loss_pseudo = _DistortionLossPseudo.apply

class _DistortionLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, midpoint, full_weight, dt):
        accum, dm, dw, dt = distortion_bidir(midpoint, full_weight, dt)

        ic(dm, dm.shape, dw, dt)
        ctx.save_for_backward(dm, dw, dt)
        return accum

    @staticmethod
    def backward(ctx, daccum):
        dm, dw, dt = ctx.saved_tensors
        return daccum * dm, daccum * dw, daccum * dt

distortion_loss = _DistortionLoss.apply

def lossfun_distortion(midpoint, full_weight, dt):
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    # extend the mipoint artifically to the background
    dut = torch.abs(midpoint[..., :, None] - midpoint[..., None, :])
    # mp = midpoint[..., None]
    # dut = torch.cdist(mp, mp, p=1)
    # loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)
    B = dt.shape[0]
    loss_inter = torch.einsum('bj,bk,bjk', full_weight.reshape(B, -1), full_weight.reshape(B, -1), dut)
    # ic(dt.shape, full_weight.shape)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(full_weight**2 * dt) / 3
    # ic(1, loss_inter, loss_intra)

    return loss_inter + loss_intra

if __name__ == "__main__":
    B = 3
    M = 16
    device = torch.device('cuda')
    midpoint = torch.rand(B, M, device=device)#.double()
    full_weight = torch.rand(B, M, device=device)#.double()
    dt = torch.rand(B, M, device=device)#.double()
    midpoint.requires_grad = True 
    full_weight.requires_grad = True 
    dt.requires_grad = True 
    print(distortion_loss_pseudo(midpoint, full_weight, dt))
    print(distortion_loss(midpoint, full_weight, dt))
    # torch.autograd.gradcheck(distortion_loss_pseudo, (midpoint, full_weight, dt))
    # torch.autograd.gradcheck(distortion_loss, (midpoint, full_weight, dt))
