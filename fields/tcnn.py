import torch
import tinycudann as tcnn
class TCNNRF(torch.nn.Module):
    def __init__(self, aabb, grid_size, *args, **kargs):
        super().__init__()
        self.separate_appgrid = False
        self.register_buffer('aabb', aabb)
        self.encoding = tcnn.Encoding(3, **kwargs)
        # The default one
        # dict(
        #     otype="HashGrid",
        #     n_levels=16,
        #     n_features_per_level=2,
        #     log2_hashmap_size=14,
        #     base_resolution=1,
        #     per_level_scale=2
        # )

    def set_smoothing(self, sm):
        pass
        
    def set_register(self, name, val):
        if hasattr(self, name):
            setattr(self, name, val.type_as(getattr(self, name)))
        else:
            self.register_buffer(name, val)
        
    def normalize_coord(self, xyz_sampled):
        coords = (xyz_sampled[..., :3]-self.aabb[0]) * self.invaabbSize - 1
        size = xyz_sampled[..., 3:4]
        normed = torch.cat((coords, size), dim=-1)
        if self.contract_space:
            r = 1
            d = 3
            dist = torch.linalg.norm(normed[..., :d], dim=-1, keepdim=True, ord=2) + 1e-8
            direction = normed[..., :d] / dist
            #  contracted = torch.where(dist > 1, (r+1)-r/dist, dist)/2
            contracted = torch.where(dist > 1, (dist-1)/4+1, dist)/2
            return torch.cat([ contracted * direction, normed[..., d:] ], dim=-1)
        else:
            return normed

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        raise Exception("Not implemented")

    def init_svd_volume(self, res, device):
        pass

    def vector_comp_diffs(self):
        raise Exception("Not implemented")

    def compute_features(self, xyz_sampled):
        raise Exception("Not implemented")

    def compute_densityfeature(self, xyz_sampled):
        raise Exception("Not implemented")

    def compute_appfeature(self, xyz_sampled):
        raise Exception("Not implemented")

    def shrink(self, new_aabb, voxel_size):
        raise Exception("Not implemented")
