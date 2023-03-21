import torch
from .tensor_base import TensorBase
from icecream import ic

class ListRF(torch.nn.Module):
    def __init__(self, rfs, offsets, aabbs, rots):
        super().__init__()
        self.rfs = torch.nn.ModuleList(rfs)
        self.register_buffer('offsets', torch.stack(offsets, dim=0))
        self.register_buffer('aabbs', torch.stack(aabbs, dim=0))
        self.register_buffer('rots', torch.stack(rots, dim=0))
        self.separate_appgrid = False
        self.contract_space = False
        self.nSamples = rfs[0].nSamples

    def get_device(self):
        return self.rfs[0].get_device()

    @property
    def distance_scale(self):
        return self.rfs[0].distance_scale

    @property
    def stepSize(self):
        return min([rf.stepSize for rf in self.rfs])

    @property
    def app_dim(self):
        return self.rfs[0].app_dim

    @property
    def units(self):
        return torch.stack([rf.units for rf in self.rfs], dim=0).min(dim=0)

    def normalize_coord(self, xyz):
        nxyz = self.rfs[0].normalize_coord(xyz)
        return nxyz

    def compute_densityfeature(self, xyz, *args, **kwargs):
        # _, masks = self.get_rf(xyz)
        # sigma = torch.zeros((xyz.shape[0],), device=xyz.device)
        sigmas = []
        for i, rf in enumerate(self.rfs):
            # sigma = sigma + rf.compute_densityfeature(xyz+self.offsets[i][:, :xyz.shape[-1]], *args, **kwargs)
            rxyz = torch.matmul(self.rots[i].reshape(1, 3, 3), xyz[:, :3].reshape(-1, 3, 1)).reshape(-1, 3)
            oxyz = torch.cat([rxyz, xyz[:, 3:]], dim=-1)+self.offsets[i][:, :xyz.shape[-1]]
            sigmas.append(rf.compute_densityfeature(oxyz, *args, **kwargs))
        # sigma = self.rfs[0].compute_densityfeature(xyz, *args, **kwargs)
        sigma = torch.stack(sigmas, dim=0).max(dim=0).values
        # sigma = sum(sigmas)
        return sigma

    def compute_feature(self, xyz, *args, **kwargs):
        # _, masks = self.get_rf(xyz)
        sigma = torch.zeros((xyz.shape[0],), device=xyz.device)
        norm = torch.zeros((xyz.shape[0],1), device=xyz.device)
        feat = torch.empty((xyz.shape[0], self.app_dim), device=xyz.device)

        feats = []
        alphas = []
        sigmas = []
        salpha = 0
        for i, rf in enumerate(self.rfs):
            rxyz = torch.matmul(self.rots[i].reshape(1, 3, 3), xyz[:, :3].reshape(-1, 3, 1)).reshape(-1, 3)
            oxyz = torch.cat([rxyz, xyz[:, 3:]], dim=-1)+self.offsets[i][:, :xyz.shape[-1]]
            sigd = rf.compute_densityfeature(oxyz, *args, **kwargs)
            ifeat = rf.compute_appfeature(oxyz, *args, **kwargs)
            aabb = self.aabbs[i]
            mask = (aabb[0].reshape(1, 3) < oxyz[:, :3]).all(dim=-1) & (aabb[1].reshape(1, 3) > oxyz[:, :3]).all(dim=-1)
            mask = torch.ones_like(mask)
            sigd = sigd*mask
            ifeat = ifeat*mask.reshape(-1, 1)
            alpha = (sigd.clip(min=-10, max=10).exp() * mask).reshape(-1, 1)
            # alpha = (sigd.clip(min=-10, max=10).exp()).reshape(-1, 1)
            sigmas.append(sigd)
            feats.append(ifeat)
            alphas.append(alpha)
            # feat  = feat + alpha*ifeat
            # sigma  = sigma + alpha.reshape(-1)*sigd
            # salpha = salpha + alpha

        # feat = feat / salpha
        # sigma = sigma / salpha.reshape(-1)
        inds = torch.stack(alphas, dim=0).max(dim=0).indices.reshape(-1)
        feat = torch.stack(feats, dim=1)[range(inds.shape[0]), inds]
        sigma = torch.stack(sigmas, dim=1)[range(inds.shape[0]), inds]
        return sigma, feat
