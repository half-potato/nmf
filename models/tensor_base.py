import torch

class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp, appearance_n_comp,
                 app_dim, step_ratio, density_res_multi, *args, **kwargs):
        super().__init__()
        self.density_n_comp = density_n_comp
        self.density_res_multi = density_res_multi
        self.app_n_comp = appearance_n_comp
        self.gridSize = gridSize
        self.app_dim = app_dim
        self.aabb = aabb
        self.device = device
        self.step_ratio = step_ratio

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.update_stepSize(gridSize)
        self.init_svd_volume(gridSize[0], device)

    def get_kwargs(self):
        return {
            'gridSize':self.gridSize.tolist(),
            'aabb': self.aabb,
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,
            'step_ratio': self.step_ratio,
            'density_res_multi': self.density_res_multi,
        }

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        print("density grid size", [int(self.density_res_multi*g) for g in gridSize])
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def normalize_coord(self, xyz_sampled):
        coords = (xyz_sampled[..., :3]-self.aabb[0]) * self.invaabbSize - 1
        size = xyz_sampled[..., 3:4]
        return torch.cat((coords, size), dim=-1)

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        raise Exception("Not implemented")

    def init_svd_volume(self, res, device):
        raise Exception("Not implemented")

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
