import torch
import torch.nn.functional as F
from icecream import ic
import numpy as np
import utils

from torch.cuda.amp import custom_fwd, custom_bwd

class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x.clamp(-15, 10))

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 10))


def normalize(v, ord=2, eps=torch.finfo(torch.float32).eps):
    if ord == 2:
        return v / (v**2).sum(axis=-1, keepdim=True).clip(min=eps).sqrt()
    else:
        return v / (torch.linalg.norm(v, dim=-1, keepdim=True, ord=ord)+1e-8)

class TensorBase(torch.nn.Module):
    def __init__(self, aabb, density_shift, activation, lr, lr_net, contract_space=False, distance_scale=25, num_pretrain=0, **kwargs):
        super().__init__()
        self.lr = lr
        self.lr_net = lr_net
        self.activation = activation
        self.num_pretrain = num_pretrain
        self.density_shift = density_shift
        self.contract_space = contract_space
        self.distance_scale = distance_scale
        self.set_aabb(aabb)

    def set_aabb(self, aabb):
        self.set_register('aabb', aabb)
        self.set_register('aabbSize', aabb[1] - aabb[0])
        self.set_register('invaabbSize', 2.0/self.aabbSize)
        self.set_register('aabbDiag', torch.sqrt(torch.sum(torch.square(self.aabbSize))))

    def get_device(self):
        return self.aabbSize.device
        
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

    def feature2density(self, density_features):
        if self.activation == "softplus":
            return F.softplus(density_features.clamp(-15, 1e3)+self.density_shift)
        elif self.activation == "relu":
            return F.relu(density_features+self.density_shift)
        elif self.activation == "exp":
            return TruncExp.apply(density_features+self.density_shift)
        elif self.activation == "identity":
            return density_features
        else:
            raise Exception (f"Unknown activation {self.activation}")

    def set_register(self, name, val):
        if hasattr(self, name):
            setattr(self, name, val.type_as(getattr(self, name)))
        else:
            self.register_buffer(name, val)

    def density_L1(self):
        raise Exception("Not implemented")

    def vector_comp_diffs(self):
        raise Exception("Not implemented")

    def compute_densityfeature(self, xyz_sampled, activate=True):
        sigfeat = self._compute_densityfeature(self.normalize_coord(xyz_sampled))
        if activate:
            return self.feature2density(sigfeat).reshape(-1)
        else:
            return sigfeat.reshape(-1)

    def compute_feature(self, xyz_sampled):
        sigfeat, feat = self._compute_feature(self.normalize_coord(xyz_sampled))
        return self.feature2density(sigfeat).reshape(-1), feat

    def compute_appfeature(self, xyz_sampled):
        return self._compute_appfeature(self.normalize_coord(xyz_sampled))

    def _compute_densityfeature(self, xyz_sampled):
        raise Exception("Not implemented")

    def _compute_feature(self, xyz_sampled):
        raise Exception("Not implemented")

    def _compute_appfeature(self, xyz_sampled):
        raise Exception("Not implemented")


class TensorVoxelBase(TensorBase):
    def __init__(self, aabb, density_n_comp, appearance_n_comp, step_ratio, 
                 app_dim, density_res_multi,
                 N_voxel_init, N_voxel_final, upsamp_list, grid_size=None, **kwargs):
        super().__init__(aabb, **kwargs)
        self.separate_appgrid = True
        self.dtype = torch.half
        self.density_n_comp = [density_n_comp]*3
        self.app_n_comp = [appearance_n_comp]*3
        self.density_res_multi = density_res_multi
        self.app_dim = app_dim
        self.step_ratio = step_ratio
        self.N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(N_voxel_init), np.log(N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]
        self.upsamp_list = upsamp_list

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        grid_size = torch.tensor(utils.N_to_reso(N_voxel_init, self.aabb)) if grid_size is None else grid_size

        self.update_stepSize(grid_size)

    def update_stepSize(self, grid_size):
        grid_size = torch.LongTensor(grid_size)
        print("grid size", grid_size)
        print("aabb", self.aabb.view(-1))

        self.set_register('grid_size', grid_size)
        self.set_register('units', self.aabbSize.to(self.grid_size.device) / (self.grid_size-1))
        # min is more accurate than mean
        self.set_register('stepSize', torch.min(self.units)*self.step_ratio)
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def check_schedule(self, iter, batch_mul):
        upsamp_list = [i*batch_mul for i in self.upsamp_list]
        if iter in upsamp_list:
            i = upsamp_list.index(iter)
            n_voxels = self.N_voxel_list[i]
            reso_cur = utils.N_to_reso(n_voxels, self.aabb)
            # nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio/tensorf.rf.density_res_multi))
            self.upsample_volume_grid(reso_cur)
            return True
        return False

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        raise Exception("Not implemented")

    def init_svd_volume(self, res, device):
        pass

    def shrink(self, new_aabb, voxel_size):
        raise Exception("Not implemented")
