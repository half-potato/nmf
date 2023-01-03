from icecream import ic
import torch
import time

# warp isn't faster
def row_mask_sum(mat, mask):
    B, M = mask.shape
    N, D = mat.shape
    device = mat.device
    dtype = mat.dtype
    full_mat = torch.zeros((B, M, D), dtype=dtype, device=device)
    full_mat[mask] = mat
    return full_mat.sum(dim=1)
"""
import warp as wp

wp.init()

@wp.kernel
def cache_mask_inds_kernel(
    mask_row_sum: wp.array(dtype=int),
    mask: wp.array2d(dtype=int),
    mask_inds: wp.array2d(dtype=int),
    M: int):
    i = wp.tid()
    start_ind = int(0)
    for ind in range(i):
        start_ind += mask_row_sum[ind]
    for j in range(M):
        if mask[i, j] != 0:
            mask_inds[i, j] = start_ind
            start_ind += 1


@wp.kernel
def row_mask_sum_kernel(
    mat: wp.array2d(dtype=float),
    mask: wp.array2d(dtype=int),
    mask_inds: wp.array2d(dtype=int),
    sum_out: wp.array2d(dtype=float),
    M: int,
    D: int):
    i = wp.tid()
    for d in range(D):
        s = float(0)
        for j in range(M):
            if mask[i, j] != 0:
                ind = mask_inds[i, j]
                s += mat[ind, d]
        sum_out[i, d] = s

def _row_mask_sum(mat, mask):
    B, M = mask.shape
    N, D = mat.shape
    device = mat.device
    device = 'cuda'
    dtype = mat.dtype
    dtype = float
    assert(mat.shape[0] == mask.sum())

    mask_row_sum_wp = wp.from_torch(mask.sum(dim=1).int(), dtype=wp.int32)
    mat_wp = wp.from_torch(mat)
    mask_wp = wp.from_torch(mask.int(), dtype=wp.int32)
    mask_inds_wp = wp.from_torch(torch.zeros((B, M), device=mat.device, dtype=torch.int32), dtype=wp.int32)
    sum_out_wp = wp.from_torch(torch.zeros((B, D), device=mat.device, dtype=mat.dtype))

    # mask_inds_wp = wp.zeros((B, M), dtype=wp.int32, device=device)
    # sum_out_wp = wp.zeros((B, D), dtype=dtype, device=device)
    wp.launch(kernel=cache_mask_inds_kernel,
              dim=(B),
              inputs=[mask_row_sum_wp, mask_wp, mask_inds_wp, M],
              device=device)

    wp.launch(kernel=row_mask_sum_kernel,
              dim=(B),
              inputs=[mat_wp, mask_wp, mask_inds_wp, sum_out_wp, M, D],
              device=device)

    sum_out = wp.to_torch(sum_out_wp)
    mask_inds = wp.to_torch(mask_inds_wp)

    return sum_out, mask_inds

@wp.kernel
def row_mask_sum_backward_kernel(
    mask: wp.array2d(dtype=int),
    mask_inds: wp.array2d(dtype=int),
    dsum_out: wp.array2d(dtype=float),
    dmat: wp.array2d(dtype=float),
    M: int,
    D: int):
    i = wp.tid()
    for j in range(M):
        if mask[i, j] != 0:
            for d in range(D):
                ind = mask_inds[i, j]
                dmat[ind, d] = dsum_out[i, d]

def _row_mask_sum_backward(N, D, mask, mask_inds, dsum_out):
    device = dsum_out.device
    device = 'cuda'
    dtype = dsum_out.dtype
    dtype = float
    B, M = mask.shape

    mask_wp = wp.from_torch(mask.int(), dtype=wp.int32)
    mask_inds_wp = wp.from_torch(mask_inds.int(), dtype=wp.int32)
    dsum_out_wp = wp.from_torch(dsum_out)
    dmat_wp = wp.from_torch(torch.zeros((N, D), device=dsum_out.device, dtype=dsum_out.dtype))
    # dmat_wp = wp.zeros((N, D), dtype=dtype, device = device)

    wp.launch(kernel=row_mask_sum_backward_kernel,
              dim=(B),
              inputs=[mask_wp, mask_inds_wp, dsum_out_wp, dmat_wp, M, D],
              device=device)

    dmat = wp.to_torch(dmat_wp)
    return dmat
    

class _RowMaskSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat, mask):
        start = time.time()
        sum_out, mask_inds = _row_mask_sum(mat, mask)
        # ic(full_weight.mean())

        # ic(dm, dw, dt)
        ctx.save_for_backward(mask, mask_inds)
        ctx.N = mat.shape[0]
        ctx.D = mat.shape[1]
        return sum_out

    @staticmethod
    def backward(ctx, dsum_out):
        mask, mask_inds = ctx.saved_tensors
        dmat = _row_mask_sum_backward(ctx.N, ctx.D, mask, mask_inds, dsum_out)
        return dmat, None

row_mask_sum = _RowMaskSum.apply


if __name__ == "__main__":
    B = 100
    M = 100
    device = torch.device('cuda')
    dtype = torch.float
    full_mat = torch.rand(B, M, 3, dtype=dtype, device=device)
    full_mat.requires_grad = True 
    mask = full_mat[..., 0] > 0.3
    mat = full_mat[mask].detach()
    mat.requires_grad = True
    full_mat2 = torch.zeros(B, M, 3, dtype=dtype, device=device)
    full_mat2[mask] = mat
    gt_sum = (full_mat2*mask.unsqueeze(-1)).sum(dim=1)
    calc_sum = row_mask_sum(mat, mask)
    ic((calc_sum- gt_sum).mean())
    ic(torch.equal(calc_sum, gt_sum))
    deriv = torch.rand_like(gt_sum)
    dmat1 = torch.autograd.grad(gt_sum, mat, deriv, allow_unused=True)
    dmat2 = torch.autograd.grad(calc_sum, mat, deriv, allow_unused=True)
    ic(torch.equal(dmat1[0], dmat2[0]))

"""
