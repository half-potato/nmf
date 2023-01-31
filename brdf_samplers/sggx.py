import torch
import math
from modules import safemath
from mutils import normalize
from .base import PseudoRandomSampler
from icecream import ic


class SGGXSampler(PseudoRandomSampler):

    def sample(self, viewdir, normal, r1, r2, ray_mask, eps=torch.finfo(torch.float32).eps, **kwargs):
        num_samples = ray_mask.shape[1]
        # viewdir: (B, 3)
        # normal: (B, 3)
        # r1, r2: B roughness values for anisotropic roughness
        device = normal.device
        B = normal.shape[0]
        eps=torch.finfo(normal.dtype).eps

        # establish basis for BRDF
        z_up = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, 3).expand(B, 3)
        x_up = torch.tensor([-1.0, 0.0, 0.0], device=device).reshape(1, 3).expand(B, 3)
        up = torch.where(normal[:, 2:3] < 0.999, z_up, x_up)
        tangent = normalize(torch.linalg.cross(up, normal))
        bitangent = normalize(torch.linalg.cross(normal, tangent))
        # B, 3, 3
        row_world_basis = torch.stack([tangent, bitangent, normal], dim=1).reshape(B, 3, 3)

        # B, 3, 3
        S_diagv = torch.stack([r1, r2, torch.ones_like(r1)], dim=-1).reshape(-1, 3)
        S_diag = torch.diag_embed(S_diagv)
        S = torch.matmul(torch.matmul(row_world_basis, S_diag), row_world_basis.permute(0, 2, 1))
        M = torch.zeros((B, 3, 3), device=device)
        tmp = (S[:, 1, 1]*S[:, 2, 2] - S[:, 1, 2]**2).clip(min=eps).sqrt()
        M[:, 0, 0] = torch.linalg.det(S).abs().sqrt() / tmp
        # checked
        inv_sqrt_Sii = 1/S[:, 2, 2].clip(min=eps).sqrt().clip(min=eps)
        # checked
        M[:, 1, 0] = -inv_sqrt_Sii*(S[:, 0, 2]*S[:, 1, 2] - S[:, 0, 1]*S[:, 2, 2])/tmp
        M[:, 1, 1] = inv_sqrt_Sii*tmp

        # checked
        M[:, 2, 0] = inv_sqrt_Sii * S[:, 0, 2]
        M[:, 2, 1] = inv_sqrt_Sii * S[:, 1, 2]
        M[:, 2, 2] = inv_sqrt_Sii * S[:, 2, 2]

        angs = self.draw(B, num_samples).to(device)

        M_mask = M.reshape(B, 1, 3, 3).expand(B, num_samples, 3, 3)[ray_mask]
        S_mask_v = S_diagv.reshape(B, 1, 3).expand(B, num_samples, 3)[ray_mask]

        # here is where things get really large
        u1 = angs[..., 0]
        u2 = angs[..., 1]

        # stretch and mask stuff to reduce memory
        # r1_mask = r1.reshape(-1, 1).expand(u1.shape)[ray_mask]
        # r2_mask = r2.reshape(-1, 1).expand(u1.shape)[ray_mask]
        row_world_basis_mask = row_world_basis.permute(0, 2, 1).reshape(B, 1, 3, 3).expand(B, num_samples, 3, 3)[ray_mask]

        u1_mask = u1[ray_mask]
        u2_mask = u2[ray_mask]

        u1sqrt = u1_mask.clip(min=eps).sqrt()
        u = (2*math.pi*u2_mask).cos() * u1sqrt
        v = (2*math.pi*u2_mask).sin() * u1sqrt
        w = (1-u**2-v**2).clip(min=eps).sqrt()

        H_l = normalize(u[:, None] * M_mask[:, 0] + v[:, None] * M_mask[:, 1] + w[:, None] * M_mask[:, 2])

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

        temp = torch.matmul(torch.matmul(H_l.reshape(-1, 1, 3), torch.diag_embed(1/S_mask_v.clip(min=eps))), H_l.reshape(-1, 3, 1))
        prob = 1 / (math.pi * (S_mask_v[:, 0] * S_mask_v[:, 1] * S_mask_v[:, 2]).clip(min=eps).sqrt().reshape(-1) * (temp.reshape(-1))**2).clip(min=eps)

        return L, row_world_basis_mask, prob

    def compute_prob(self, halfvec, eN, r1, r2, **kwargs):
        eps=torch.finfo(halfvec.dtype).eps
        S_diag = torch.stack([r1.clip(min=eps), r2.clip(min=eps), torch.ones_like(r1)], dim=-1).reshape(-1, 3)
        temp = torch.matmul(torch.matmul(halfvec.reshape(-1, 1, 3), torch.diag_embed(1/S_diag)), halfvec.reshape(-1, 3, 1))
        prob = 1 / (math.pi * (S_diag[:, 0] * S_diag[:, 1] * S_diag[:, 2]).clip(min=eps).sqrt().reshape(-1) * (temp.reshape(-1))**2).clip(min=eps)
        return prob
