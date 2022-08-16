import torch
from icecream import ic
import triton
import triton.language as tl

@triton.jit
def distortion_bidir_kernel(
        full_weight_ptr,
        accum_ptr,
        M: tl.constexpr,
        stridem: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr):

    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_a = offs_m * stridem
    
    mask_a = (offs_m < M)

    # tl.store(dm_ptr + offs_a, offs_a*0+pid_b, mask_a)

    w1 = tl.load(full_weight_ptr + offs_a)
    print(w1)
    # wm = tl.dot(tl.reshape(w1, (BLOCK_SIZE_M, 1)), tl.reshape(w1, (1, BLOCK_SIZE_M)))
    wm = tl.reshape(w1, (BLOCK_SIZE_M, 1)) * tl.reshape(w1, (1, BLOCK_SIZE_M))
    inter_loss = tl.sum(tl.ravel(wm), axis=0)
    # inter_loss = tl.sum(tl.ravel(w1), axis=0)
    tl.atomic_add(accum_ptr, inter_loss)

def distortion_bidir(x):
    B, M = x.shape
    BLOCK_SIZE_M = 128
    device = x.device
    dtype = x.dtype
    accum = torch.zeros((1), dtype=dtype, device=device)

    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)

    distortion_bidir_kernel[grid](x, accum, M, x.stride(1), BLOCK_SIZE_M)
    return accum

device = torch.device('cuda')
fw = torch.rand((3, 256), device=device)
print(distortion_bidir(fw))
