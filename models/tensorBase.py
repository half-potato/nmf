import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time
from icecream import ic

from . import render_modules
import plotly.express as px
import plotly.graph_objects as go

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([
        torch.ones(alpha.shape[0], 1, device=alpha.device),
        1. - alpha + 1e-10
    ], dim=-1), dim=-1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0], enable_reflection = True,
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus', bundle_size = 3, density_grid_dims=8):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device
        self.enable_reflection = enable_reflection

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.bundle_size = bundle_size
        self.density_grid_dims = density_grid_dims

        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.f_blur = torch.tensor([1, 2, 1], device=device) / 4
        self.f_edge = torch.tensor([-1, 0, 1], device=device) / 2

        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        self.has_grid = False
        if shadingMode == 'MLP_PE':
            self.renderModule = render_modules.MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = render_modules.MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'BundleMLP_Fea':
            self.renderModule = render_modules.BundleMLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = render_modules.MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'BundleMLP':
            self.renderModule = render_modules.BundleMLPRender(self.app_dim, view_pe, featureC, self.bundle_size).to(device)
        elif shadingMode == 'BundleMLP_Fea_Grid':
            self.renderModule = render_modules.BundleMLPRender_Fea_Grid(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size, extra=self.density_grid_dims).to(device)
            self.reflectionModule = render_modules.BundleMLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size).to(device)
            self.has_grid = True
        elif shadingMode == 'BundleMLPSphEncGrid':
            self.renderModule = render_modules.BundleMLPRender_Fea_Grid(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size, extra=self.density_grid_dims).to(device)
            self.reflectionModule = render_modules.BundleSphEncoding(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size).to(device)
            # self.reflectionModule = render_modules.BundleDirectSphEncoding(self.app_dim, view_pe, fea_pe, featureC, self.bundle_size).to(device)
            self.has_grid = True
        elif shadingMode == 'SH':
            self.renderModule = render_modules.SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = render_modules.RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe, "density_grid_dims", self.density_grid_dims)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,
            'bundle_size': self.bundle_size,
            'density_grid_dims': self.density_grid_dims,
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])


    def compute_normal(self, depth_map, focal):
        # Compute normal map
        dy = (depth_map * (self.f_blur[None, :] * self.f_edge[:, None]).reshape(1, 3, 3)).sum(1).sum(1)
        dx = (depth_map * (self.f_blur[:, None] * self.f_edge[None, :]).reshape(1, 3, 3)).sum(1).sum(1)

        dx = dx * focal * 2 / depth_map[..., self.bundle_size//2, self.bundle_size//2]
        dy = dy * focal * 2 / depth_map[..., self.bundle_size//2, self.bundle_size//2]
        inv_denom = 1 / torch.sqrt(1 + dx**2 + dy**2)
        # (N, 3)
        normal_map = torch.stack([dx * inv_denom, -dy * inv_denom, inv_denom], -1)
        return normal_map

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            l = torch.rand_like(interpx)
            interpx += l.to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1].to(rays_o) - rays_o) / vec
        rate_b = (self.aabb[0].to(rays_o) - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples, device=rays_o.device)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            # N, N_samples
            # add noise along each ray
            brng = rng.reshape(-1, N_samples)
            # brng = brng + torch.rand_like(brng[:, [0], [0]])
            # r = torch.rand_like(brng[:, 0:1, 0:1])
            r = torch.rand_like(brng[:, 0:1])
            brng = brng + r
            rng = brng.reshape(-1, N_samples)
        step = stepsize * rng
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox


    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize
        gridSize *= 2

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1].to(rays_o) - rays_o) / vec
                rate_b = (self.aabb[0].to(rays_o) - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered], mask_filtered


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha


    def forward(self, rays_chunk, focal, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        # rays_chunk: (N, 6)

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        # xyz_sampled.shape: (N, N_samples, 3)
        # z_vals.shape: (N, N_samples)
        # ray_valid.shape: (N, N_samples)

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        rays_up = rays_chunk[:, 6:9]
        rays_o = rays_chunk[:, :3]
        rays_up = rays_up.view(-1, 1, 3).expand(xyz_sampled.shape)
        n_samples = xyz_sampled.shape[1]
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid
            all_alphas = self.alphaMask.sample_alpha(xyz_sampled).detach().cpu().reshape(xyz_sampled.shape[0], xyz_sampled.shape[1])


        # sigma.shape: (N, N_samples)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        normal = torch.zeros(xyz_sampled.shape, device=xyz_sampled.device)
        all_sigma_feature = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], self.bundle_size, self.bundle_size, 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature, normal_feature = self.compute_density_norm(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            all_sigma_feature[ray_valid] = sigma_feature
            sigma[ray_valid] = validsigma
            normal[ray_valid] = normal_feature

        # xyz_c = xyz_sampled.detach().cpu()
        # fig = px.scatter_3d(x=xyz_c[:64, :, 0].flatten(), y=xyz_c[:64, :, 1].flatten(), z=xyz_c[:64, :, 2].flatten(), color=sigma.detach().cpu()[:64].flatten())
        # fig.show()
        # assert(False)


        # weight: [N_rays, N_samples]
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        # app stands for appearance
        app_mask = weight > self.rayMarch_weight_thres
        rgb_app_mask = app_mask[:, :, None, None].repeat(1, 1, self.bundle_size, self.bundle_size)
        bundle_weight = weight[..., None, None].repeat(1, 1, self.bundle_size, self.bundle_size)
        bundle_size_w = z_vals / focal * (self.bundle_size-1)

        normal_map = torch.sum(weight[..., None] * normal, 1)
        world_normal = normal_map
        refdirs = viewdirs - 2 * (viewdirs * world_normal[:, None]).sum(-1, keepdim=True) * world_normal[:, None]

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            if self.has_grid:
                # bundle_size_w: (N, 1)
                valid_rgbs, rel_density_grid_feature, roughness = self.renderModule(xyz_sampled[app_mask], refdirs[app_mask], app_features, bundle_size_w[app_mask][:, None], rays_up[app_mask])

                # [N_rays, N_samples, bundle_size, bundle_size]
                # bundle_sigma = sigma[..., None, None].repeat(1, 1, self.bundle_size, self.bundle_size)
                bundle_sigma = all_sigma_feature[..., None, None].repeat(1, 1, self.bundle_size, self.bundle_size)
                # ic(1, bundle_sigma[app_mask], bundle_sigma[app_mask].shape, rel_density_grid_feature.shape)

                # bundle_sigma[app_mask] = F.softplus(bundle_sigma[app_mask] + rel_density_grid_feature, beta=10)
                # bundle_sigma[app_mask] = F.softplus(bundle_sigma[app_mask], beta=10)
                # bundle_sigma[app_mask] = F.relu(bundle_sigma[app_mask] + rel_density_grid_feature)

                bundle_sigma[app_mask] += rel_density_grid_feature
                bundle_sigma[ray_valid] = self.feature2density(bundle_sigma[ray_valid])

                # ic(2, bundle_sigma[app_mask], rel_density_grid_feature)
                # ic(bundle_weight[app_mask][:10], rel_density_grid[:10], rel_density, extra)

                # reshape to align with how raw2alpha expects
                bundle_sigma = bundle_sigma.permute(0, 2, 3, 1).reshape(-1, n_samples)
                bundle_dists = dists.reshape(-1, 1, n_samples).repeat(1, self.bundle_size**2, 1).reshape(-1, n_samples)
                _, bundle_weight, _ = raw2alpha(bundle_sigma, bundle_dists * self.distance_scale)
                # reshape back to original
                bundle_weight = bundle_weight.reshape(-1, self.bundle_size, self.bundle_size, n_samples).permute(0, 3, 1, 2)
            else:
                valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[rgb_app_mask] = valid_rgbs.reshape(-1, 3)

        acc_map = torch.sum(bundle_weight, 1)

        # calculate depth
        bundle_z_vals = z_vals[..., None, None].repeat(1, 1, self.bundle_size, self.bundle_size)
        bundle_rays_chunk = rays_chunk[:, None, None, :].repeat(1, self.bundle_size, self.bundle_size, 1)
        depth_map = torch.sum(bundle_weight * bundle_z_vals, 1)
        # (N, bundle_size, bundle_size)
        depth_map = depth_map + (1. - acc_map) * bundle_rays_chunk[..., -1]

        # normal_map = self.compute_normal(depth_map, focal)

        inds = bundle_weight[:, :, self.bundle_size//2, self.bundle_size//2].max(dim=1).indices
        xyz = xyz_sampled[range(xyz_sampled.shape[0]), inds]#.cpu().numpy()
        # ref_mask = reflectivity > 0.1
        if self.enable_reflection and self.bundle_size == 3 and app_mask.any() and False:

            # project normal map from camera space to world space
            # z_basis = -torch.cross(-viewdirs[:, 0], rays_up[:, 0])
            # world_normal.shape: (N, 3)
            # world_normal = normal_map[..., 0:1] * z_basis + normal_map[..., 1:2] * rays_up[:, 0] + normal_map[..., 2:3] * -viewdirs[:, 0]
            world_normal = normal_map
            # world_normal = -viewdirs[:, 0]
            # world_normal = normal_map
            # world_normal = rays_up[:, 0]
            # world_normal = normal_map[..., 1:2] * rays_up[:, 0]

            # ic(rays_up[:, 0].max(dim=0), rays_up[:, 0].min(dim=0))
            # ic(viewdirs[:, 0].max(dim=0), viewdirs[:, 0].min(dim=0))

            # right_vec = (z_basis).cpu()
            # up_vec = (rays_up[:, 0]).cpu()
            # front_vec = (-viewdirs[:, 0]).cpu()
            # v_world_normal = world_normal.cpu()
            # v_normal_map = normal_map.cpu()

            # g1 = go.Scatter3d(x=right_vec[:, 0], y=right_vec[:, 1], z=right_vec[:, 2], marker=dict(color='red'), mode='markers')
            # g3 = go.Scatter3d(x=front_vec[:, 0], y=front_vec[:, 1], z=front_vec[:, 2], marker=dict(color='blue'), mode='markers')
            # g2 = go.Scatter3d(x=up_vec[:, 0], y=up_vec[:, 1], z=up_vec[:, 2], marker=dict(color='green'), mode='markers')
            # g4 = go.Scatter3d(x=v_world_normal[:, 0], y=v_world_normal[:, 1], z=v_world_normal[:, 2], marker=dict(color='purple'), mode='markers')
            # g5 = go.Scatter3d(x=v_normal_map[:, 0], y=v_normal_map[:, 1], z=v_normal_map[:, 2], marker=dict(color='orange'), mode='markers')

            # fig = go.Figure(data=[g1, g2, g3, g4, g5])
            # fig.show()
            # assert(False)

            # reflected vector: r = d - 2(d*n)n
            # d = (0, 0, 1) relative to the normal
            refdirs = viewdirs - 2 * (viewdirs * world_normal[:, None]).sum(-1, keepdim=True) * world_normal[:, None]

            ref_valid_rgbs = self.reflectionModule(xyz_sampled[app_mask], refdirs[app_mask], app_features, bundle_size_w[app_mask][:, None], rays_up[app_mask], roughness)
            rgb[rgb_app_mask] += ref_valid_rgbs.reshape(-1, 3)
        else:
            world_normal = normal_map

        rgb_map = torch.sum(bundle_weight[..., None] * rgb, 1)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        return rgb_map, depth_map, normal_map, acc_map, xyz
        # return rgb_map, depth_map, world_normal, acc_map, xyz

