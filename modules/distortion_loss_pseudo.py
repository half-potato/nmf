import torch
from icecream import ic

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

def lossfun_distortion2(t, w, dt):
    device = w.device
    B, n_samples = w.shape
    full_weight = torch.cat([w, 1-w.sum(dim=1, keepdim=True)], dim=1)
    #
    # midpoint = t
    # fweight = torch.abs(midpoint[..., :, None] - midpoint[..., None, :])
    # # # ut = (z_vals[:, 1:] + z_vals[:, :-1])/2
    # #
    # loss_inter = torch.einsum('bj,bk,jk', full_weight.reshape(B, -1), full_weight.reshape(B, -1), fweight)
    # loss_intra = (w**2 * dt).sum(dim=1).sum()/3
    #
    # # this one consumes too much memory

    S = torch.linspace(0, 1, n_samples+1, device=device).reshape(-1, 1)
    # S = t[0, :].reshape(-1, 1)
    fweight = (S - S.T).abs()
    # ut = (z_vals[:, 1:] + z_vals[:, :-1])/2

    floater_loss_1 = torch.einsum('bj,bk,jk', full_weight.reshape(B, -1), full_weight.reshape(B, -1), fweight)
    floater_loss_2 = (full_weight**2).sum()/3/n_samples
    # ic(fweight)

    # ic(floater_loss_1, floater_loss_2)
    return floater_loss_1 + floater_loss_2

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
                accum += (dut.abs()*wm).sum()

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
            ddt[b, m : m+BLOCK_SIZE_M ] += w1**2 / 3
            dw[b, m : m+BLOCK_SIZE_M] += 2 * w1 * pdt / 3
    return accum, dm, dw, ddt

class _DistortionLossPseudo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, midpoint, full_weight, dt):
        accum, dm, dw, dt = distortion_bidir_pseudo(midpoint, full_weight, dt)
        # ic(dm, dw, dt)

        ctx.save_for_backward(dm, dw, dt)
        return accum

    @staticmethod
    def backward(ctx, daccum):
        dm, dw, dt = ctx.saved_tensors
        return daccum * dm, daccum * dw, daccum * dt

distortion_loss_pseudo = _DistortionLossPseudo.apply

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
    dtype = torch.double
    midpoint = torch.rand(B, M, dtype=dtype, device=device)
    full_weight = torch.rand(B, M, dtype=dtype, device=device)
    dt = torch.rand(B, M, dtype=dtype, device=device)
    midpoint.requires_grad = True 
    full_weight.requires_grad = True 
    dt.requires_grad = True 
    print(distortion_loss_pseudo(midpoint, full_weight, dt))
    torch.autograd.gradcheck(distortion_loss_pseudo, (midpoint, full_weight, dt))
    # torch.autograd.gradcheck(distortion_loss, (midpoint, full_weight, dt))
