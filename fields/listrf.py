import torch
from .tensor_base import TensorBase
from models.tensor_nerf import TensorNeRF

class ListRF(TensorBase):
    def __init__(self, rfs, offsets):
        self.rfs = rfs
        self.offsets = offsets
        self.separate_appgrid = False
        self.contract_space = False

    @staticmethod
    def load(paths, offsets, **kwargs):
        rfs = []
        for path in paths:
            ckpt = torch.load(path)
            tensorf = TensorNeRF.load(ckpt, **kwargs)
            rfs.append(tensorf)
        return ListRF(paths, offsets)

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

    def get_rf(self, xyz):
        # xyz: (-1, 3)
        # returns: list of points and a list of corresponding mask to reassemble the original list
        pts = []
        masks = []
        for rf, offset in zip(self.rfs, self.offsets):
            bound = (rf.aabb[0].reshape(1, 3) < (xyz + offset)) & (rf.aabb[1].reshape(1, 3) > (xyz + offset))
            mask = bound[:, 0] & bound[:, 1] & bound[:, 2]
            pts.append(xyz[mask], mask)
        return pts, masks

    def normalize_coord(self, xyz):
        pts, masks = self.get_rf(xyz)
        nxyz = torch.empty_like(xyz)
        for rf, offset, mpts, mask in zip(self.rfs, self.offsets, pts, masks):
            nxyz[mask] = rf.normalize_coord(mpts + offset)
        return nxyz

    def compute_densityfeature(self, xyz):
        _, masks = self.get_rf(xyz)
        nxyzs = self.normalize_coord(xyz)
        sigma = torch.empty_like(xyz[:, 0])
        for rf, mask, nxyz in zip(self.rfs, masks, nxyzs):
            sigma[mask] = rf.compute_densityfeature(nxyz)
        return sigma

    def compute_appfeature(self, xyz):
        _, masks = self.get_rf(xyz)
        nxyzs = self.normalize_coord(xyz)
        sigma = torch.empty_like(xyz[:, 0])
        for rf, mask, nxyz in zip(self.rfs, masks, nxyzs):
            sigma[mask] = rf.compute_densityfeature(nxyz)
        return sigma
