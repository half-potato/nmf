import torch
from .tensor_base import TensorBase
from icecream import ic

class ListRF(torch.nn.Module):
    def __init__(self, rfs, offsets):
        super().__init__()
        self.rfs = torch.nn.ModuleList(rfs)
        self.register_buffer('offsets', torch.stack(offsets, dim=0))
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
        sigma = torch.zeros((xyz.shape[0],), device=xyz.device)
        for i, rf in enumerate(self.rfs):
            sigma = sigma + rf.compute_densityfeature(xyz+self.offsets[i][:, :xyz.shape[-1]], *args, **kwargs)
        # sigma = self.rfs[0].compute_densityfeature(xyz, *args, **kwargs)
        return sigma / len(self.rfs)

    def compute_appfeature(self, xyz, *args, **kwargs):
        # _, masks = self.get_rf(xyz)
        sigma = torch.zeros((xyz.shape[0],), device=xyz.device)
        norm = torch.zeros((xyz.shape[0],1), device=xyz.device)
        feat = torch.empty((xyz.shape[0], self.app_dim), device=xyz.device)

        for i, rf in enumerate(self.rfs):
            sigd = rf.compute_densityfeature(xyz+self.offsets[i][:, :xyz.shape[-1]], *args, **kwargs)
            ifeat = rf.compute_appfeature(xyz+self.offsets[i][:, :xyz.shape[-1]], *args, **kwargs)
            alpha = sigd.exp().reshape(-1, 1)
            sigma = sigma + sigd
            feat  = feat + alpha*ifeat
            norm = norm + alpha
        # sigma = self.rfs[0].compute_densityfeature(xyz, *args, **kwargs)
        return feat / norm.clip(min=1e-8)

    def compute_feature(self, xyz, *args, **kwargs):
        # _, masks = self.get_rf(xyz)
        sigma = torch.zeros((xyz.shape[0],), device=xyz.device)
        norm = torch.zeros((xyz.shape[0],1), device=xyz.device)
        feat = torch.empty((xyz.shape[0], self.app_dim), device=xyz.device)

        feats = []
        alphas = []
        for i, rf in enumerate(self.rfs):
            sigd = rf.compute_densityfeature(xyz+self.offsets[i][:, :xyz.shape[-1]], *args, **kwargs)
            ifeat = rf.compute_appfeature(xyz+self.offsets[i][:, :xyz.shape[-1]], *args, **kwargs)
            alpha = sigd.clip(min=-10, max=10).exp().reshape(-1, 1)
            sigma = sigma + sigd
            # feat  = feat + alpha*ifeat
            feats.append(ifeat)
            alphas.append(alpha)

        inds = torch.stack(alphas, dim=0).max(dim=0).indices.reshape(-1)
        feat = torch.stack(feats, dim=1)[range(inds.shape[0]), inds]
        # sigma = self.rfs[0].compute_densityfeature(xyz, *args, **kwargs)
        return sigma, feat
