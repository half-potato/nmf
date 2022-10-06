import torch
from mutils import normalize
import warp as wp
from modules.distortion_loss_warp import from_torch
from icecream import ic
import random

wp.init()
@wp.kernel
def sample_bright_spots_kernel(
    Vs: wp.array2d(dtype=float),
    Ns: wp.array2d(dtype=float),
    spots: wp.array2d(dtype=float),
    num_ele: wp.array(dtype=int),
    num_ray: wp.array(dtype=int),
    B: int,
    m: int,
    Ls_out: wp.array3d(dtype=float),
    bright_mask_out: wp.array2d(dtype=int),
    std: float,
    rand_seed: wp.uint32
):
    ti = wp.tid()
    for _i in range(m):
        i = ti*m + _i
        if i < B:
            n = num_ele[i]
            start_ind = num_ray[i]-1
            # start_ind = 0
            V = Vs[i]
            N = Ns[i]
            for j in range(n):
                dir = spots[j]
                x = dir[0] + std*wp.randf(rand_seed)
                y = dir[1] + std*wp.randf(rand_seed)
                z = dir[2] + std*wp.randf(rand_seed)
                # LdotV = x*V[0] + y*V[1] + z*V[2]
                LdotN = x*N[0] + y*N[1] + z*N[2]
                if LdotN > 0.0:
                    Ls_out[i, start_ind-j, 0] = x
                    Ls_out[i, start_ind-j, 1] = y
                    Ls_out[i, start_ind-j, 2] = z
                    bright_mask_out[i, start_ind-j] = 1

def make_side_basis(i, j, v1, v2, v3):
    k = list(set([0, 1, 2]).difference(set([i, j])))[0]
    b = torch.zeros((3, 3))
    b[i, 0] = v1
    b[j, 1] = v2
    b[k, 2] = v3
    return b

class BrightnessImportanceSampler(torch.nn.Module):
    def __init__(self, cold_start_bg_iters, scale, max_samples, update_freq):
        super().__init__()
        self.scale = scale
        self.cold_start_bg_iters = cold_start_bg_iters
        self.update_freq = update_freq
        self.max_samples = max_samples
        self.spots = None
        cubemap_basis = torch.stack([
            make_side_basis(1, 2, -1, -1, 1), # +x, x+ maps to z-, y+ maps to y+
            make_side_basis(1, 2, -1, 1, -1), # -x, x+ maps to z+, y+ maps to y+
            make_side_basis(2, 0, 1, 1, 1), # +y, x+ maps to x+, y+ maps to z+
            make_side_basis(2, 0, -1, 1, -1), # -y, x+ maps to x+, y+ maps to z+
            make_side_basis(1, 0, -1, 1, 1), # +z, x+ maps to x+, y+ maps to y+
            make_side_basis(1, 0, -1, -1, -1), # -z, x+ maps to x-, y+ maps to y+
        ], dim=0)
        self.register_buffer("cubemap_basis", cubemap_basis)

    def is_initialized(self):
        return self.spots is not None

    def inverse_index(self, face_ind, ij, res):
        device = self.cubemap_basis.device
        ij = ij.to(device)
        xy1 = (res/2 - (res/2-0.5 - ij)).trunc()
        xy = -(res - 2*xy1-1)/res
        xyz = torch.bmm(self.cubemap_basis[face_ind], torch.cat([xy, torch.ones_like(xy[:, 0:1])], dim=1).reshape(-1, 3, 1)).reshape(-1, 3)
        # xyz[:, 1] += 0.3
        # xyz[:, 2] *= 0
        # ic(xyz, self.cubemap_basis[face_ind], xy)
        # return xyz
        return normalize(xyz)

    def update(self, bg_module):
        face_ind, ij, pix_size = bg_module.get_bright_spots(self.scale, 2*self.max_samples)
        # convert cube map indices to vectors
        # convert pix size to standard deviation on sphere
        # spots are from least to greatest
        self.spots = self.inverse_index(face_ind, ij, bg_module.bg_resolution//self.scale).flip(dims=[0]).reshape(-1, 3)
        self.pix_size = pix_size
        # ic(xy.max())
        # ic(self.spots)
        # ic(bg_module(self.spots, -100*torch.ones_like(self.spots[:, 0:1])))
        
    def check_schedule(self, iter, batch_mul, bg_module):
        if iter % (self.update_freq*batch_mul) == 0 and iter > 0 and iter > self.cold_start_bg_iters:
            self.update(bg_module)

    def sample(self, V, N, ray_mask, bright_mask):
        # generate for each, then shift according to number of elements in bright mask
        # then pad and shift again to meet the end point
        device = V.device
        B = bright_mask.shape[0]
        m = 1024
        wVs = from_torch(V)
        wNs = from_torch(N)
        wspots = from_torch(self.spots)

        Ls = torch.zeros((*bright_mask.shape, 3), device=device)
        wLs = from_torch(Ls)
        wnum_ele = from_torch(bright_mask.sum(dim=1).int(), dtype=wp.int32)
        wnum_ray = from_torch(ray_mask.sum(dim=1).int(), dtype=wp.int32)
        bmask = torch.zeros_like(bright_mask).int()
        wbmask = from_torch(bmask, dtype=wp.int32)

        seed = wp.rand_init(random.randrange(9999999))
        wp.launch(kernel=sample_bright_spots_kernel,
                  dim=(B),
                  inputs=[wVs, wNs, wspots, wnum_ele, wnum_ray, B, m, wLs, wbmask, self.pix_size, seed],
                  device='cuda')
        return Ls, bmask.bool()

if __name__ == "__main__":
    # test inverse
    device = torch.device('cuda')
    from models.bg_modules import HierarchicalCubeMap
    from pathlib import Path
    sampler = BrightnessImportanceSampler(cold_start_bg_iters=10, scale=1, max_samples=1, update_freq=10).to(device)
    for res in [4, 8, 16, 32, 64]:
        ic(res)
        for fi in range(6):
            bg_module = HierarchicalCubeMap(bg_resolution=res, num_levels=1, featureC=128, activation='softplus', power=2, lr=1e-2).to(device)
            i = res // 2 + 1
            j = res // 2 + 1
            i = 3
            j = 3
            # i = res - 1
            # j = res - 1
            bg_module.bg_mats[0].data[0, fi, i, j] = 5
            # bg_module.save(Path("./"))
            if False:
                xyz = sampler.inverse_index(torch.tensor(fi).reshape(1), torch.tensor([i, j]).reshape(1, 2), res)
                ic(xyz)
                col = bg_module(xyz.reshape(1, 3), torch.tensor(-100.0, device=device).reshape(1, 1))
                ic(col)
                # for x in torch.linspace(-1, 1, 21):
                #     for y in torch.linspace(-1, 1, 21):
                #         xyz = sampler.inverse_index(torch.tensor(fi).reshape(1), torch.tensor([i+x, j+y]).reshape(1, 2)/res*2-1, res)
                #         col = bg_module(xyz.reshape(1, 3), torch.tensor(-100.0, device=device).reshape(1, 1))
                #         if col.mean() > 0.5:
                #             ic(xyz, col, x, y)
                rng = torch.linspace(-1, 1, 100, device=device)
                xyz = normalize(torch.stack(torch.meshgrid(rng, rng, rng, indexing='ij'), dim=-1).reshape(-1, 3))
                col = bg_module(xyz, -100*torch.ones_like(xyz[:, 0]))
                ind = col[:, :1].max(dim=0).indices
                ic(xyz[ind], xyz[ind] / xyz[ind].abs().max())
                ic(col[ind])

            sampler.update(bg_module)
