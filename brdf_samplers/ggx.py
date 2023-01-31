import torch
import math
from modules import safemath
from mutils import normalize
from .base import PseudoRandomSampler
from icecream import ic

class GGXSampler(PseudoRandomSampler):

    def sample(self, viewdir, normal, r1, r2, ray_mask, eps=torch.finfo(torch.float32).eps, **kwargs):
        num_samples = ray_mask.shape[1]
        # no non isotropic stuff
        r2 = r1
        # viewdir: (B, 3)
        # normal: (B, 3)
        # r1, r2: B roughness values for anisotropic roughness
        device = normal.device
        B = normal.shape[0]

        # establish basis for BRDF
        z_up = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, 3).expand(B, 3)
        x_up = torch.tensor([-1.0, 0.0, 0.0], device=device).reshape(1, 3).expand(B, 3)
        up = torch.where(normal[:, 2:3] < 0.999, z_up, x_up)
        tangent = normalize(torch.linalg.cross(up, normal))
        bitangent = normalize(torch.linalg.cross(normal, tangent))
        # B, 3, 3
        row_world_basis = torch.stack([tangent, bitangent, normal], dim=1).reshape(B, 3, 3)


        # GGXVNDF
        # V_l = torch.matmul(torch.inverse(row_world_basis.permute(0, 2, 1)), viewdir.unsqueeze(-1)).squeeze(-1)
        # ic((normal*viewdir).sum(dim=-1).min(), (normal*viewdir).sum(dim=-1).max())
        # ic(1, V_l.min(dim=0), V_l.max(dim=0))
        V_l = torch.matmul(row_world_basis, viewdir.unsqueeze(-1)).squeeze(-1)
        # ic(2, V_l.min(dim=0), V_l.max(dim=0))
        r1_c = r1.squeeze(-1)
        r2_c = r2.squeeze(-1)
        V_stretch = normalize(torch.stack([r1_c*V_l[..., 0], r2_c*V_l[..., 1], V_l[..., 2]], dim=-1)).unsqueeze(1)
        T1 = torch.where(V_stretch[..., 2:3] < 0.999, normalize(torch.linalg.cross(V_stretch, z_up.unsqueeze(1), dim=-1)), x_up.unsqueeze(1))
        T2 = normalize(torch.linalg.cross(T1, V_stretch, dim=-1))
        z = V_stretch[..., 2].reshape(-1, 1)
        a = (1 / (1+z.detach()).clip(min=1e-5)).clip(max=1e4)
        angs = self.draw(B, num_samples).to(device)

        # here is where things get really large
        u1 = angs[..., 0]
        u2 = angs[..., 1]

        # stretch and mask stuff to reduce memory
        a_mask = a.expand(u1.shape)[ray_mask]

        r_mask_u1 = r1_c.reshape(-1, 1).expand(u1.shape)[ray_mask]
        r_mask1 = r_mask_u1
        r_mask_u2 = r2_c.reshape(-1, 1).expand(u1.shape)[ray_mask]
        r_mask2 = r_mask_u2

        z_mask = z.expand(u1.shape)[ray_mask]
        u1_mask = u1[ray_mask]
        u2_mask = u2[ray_mask]
        T1_mask = T1.expand(-1, num_samples, 3)[ray_mask]
        T2_mask = T2.expand(-1, num_samples, 3)[ray_mask]
        V_stretch_mask = V_stretch.expand(-1, num_samples, 3)[ray_mask]
        row_world_basis_mask = row_world_basis.permute(0, 2, 1).reshape(B, 1, 3, 3).expand(B, num_samples, 3, 3)[ray_mask]

        r = torch.sqrt(u1_mask)
        phi = torch.where(u2_mask < a_mask, u2_mask/a_mask*math.pi, (u2_mask-a_mask)/(1-a_mask)*math.pi + math.pi)
        P1 = (r*safemath.safe_cos(phi)).unsqueeze(-1)
        P2 = (r*safemath.safe_sin(phi)*torch.where(u2_mask < a_mask, torch.tensor(1.0, device=device), z_mask)).unsqueeze(-1)
        # ic((1-a).min(), a.min(), a.max(), phi.min(), phi.max(), (1-a).max())
        N_stretch = P1*T1_mask + P2*T2_mask + (1 - P1*P1 - P2*P2).clip(min=eps).sqrt() * V_stretch_mask
        # H_l = normalize(torch.stack([r_mask1*N_stretch[..., 0], r_mask2*N_stretch[..., 1], N_stretch[..., 2].clip(min=0)], dim=-1))
        H_l = normalize(torch.stack([r_mask1*N_stretch[..., 0], r_mask2*N_stretch[..., 1], N_stretch[..., 2]], dim=-1))

        first = torch.zeros_like(ray_mask)
        first[:, 0] = True
        H_l[first[ray_mask], 0] = 0
        H_l[first[ray_mask], 1] = 0
        H_l[first[ray_mask], 2] = 1

        H = torch.matmul(row_world_basis_mask, H_l.unsqueeze(-1)).squeeze(-1)
        # H = torch.einsum('bni,bij->bnj', H_l, row_world_basis)

        V = viewdir.unsqueeze(1).expand(-1, num_samples, 3)[ray_mask]
        # N = normal.reshape(-1, 1, 3).expand(-1, num_samples, 3)[ray_mask]
        L = (2.0 * (V * H).sum(dim=-1, keepdim=True) * H - V)

        prob = -(math.pi * r_mask1 * r_mask2 * (
            H_l[:, 0]**2 / (r_mask1**2).clip(min=eps) + 
            H_l[:, 1]**2 / (r_mask2**2).clip(min=eps) + 
            H_l[:, 2]**2
            )**2).clip(min=eps).log()

        return L, row_world_basis_mask, prob

    def compute_prob(self, halfvec, eN, r1, r2, **kwargs):
        eps=torch.finfo(torch.float32).eps
        NdotH = ((halfvec * eN).sum(dim=-1)).abs().clip(min=eps, max=1).reshape(-1, 1)
        # ic(NdotH.shape, ea.shape, brdf_weight.shape)
        logD = 2*torch.log(r1.clip(min=eps)) - 2*torch.log((NdotH**2*(r1**2-1)+1).clip(min=eps))
        return logD.exp()

    def calculate_mipval(self, H, V, N, ray_mask, r1, r2, eps: float =torch.finfo(torch.float32).eps, **kwargs):
        num_samples = ray_mask.shape[1]
        NdotH = ((H * N).sum(dim=-1)).abs().clip(min=eps, max=1)
        HdotV = (H * V).sum(dim=-1).abs().clip(min=eps, max=1)
        NdotV = (N * V).sum(dim=-1).abs().clip(min=eps, max=1)
        logD = 2*torch.log(r1.clip(min=eps)) - 2*torch.log((NdotH**2*(r1**2-1)+1).clip(min=eps))
        # ic(NdotH.shape, NdotH, D, D.mean())
        # px.scatter(x=NdotH[0].detach().cpu().flatten(), y=D[0].detach().cpu().flatten()).show()
        # assert(False)
        # ic(NdotH.mean())
        lpdf = logD + torch.log(HdotV) - torch.log(NdotV)# - torch.log(roughness.clip(min=1e-5))
        # pdf = D * HdotV / NdotV / roughness.reshape(-1, 1)
        # pdf = NdotH / 4 / HdotV
        # pdf = D# / NdotH
        indiv_num_samples = ray_mask.sum(dim=1, keepdim=True).expand(-1, num_samples)[ray_mask]
        mipval = -torch.log(indiv_num_samples.clip(min=1)) - lpdf
        return mipval
