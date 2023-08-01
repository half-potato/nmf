import math

import torch
from icecream import ic

from modules import safemath
from mutils import normalize, signed_clip

from .base import PseudoRandomSampler


def coordinate_frame(normal):
    device = normal.device
    eps = torch.finfo(normal.dtype).eps
    B = normal.shape[0]

    nx = normal[..., 0:1]
    ny = normal[..., 1:2]
    nz = normal[..., 2:3]
    a = 1 / (1 + nz).clip(min=eps)
    b = -nx * ny * a
    flat_shape = [1] * (len(normal.shape) - 1) + [3]
    vecx = (
        torch.tensor([0.0, -1.0, 0.0], device=device)
        .reshape(flat_shape)
        .expand(normal.shape)
    )
    vecy = (
        torch.tensor([-1.0, 0.0, 0.0], device=device)
        .reshape(flat_shape)
        .expand(normal.shape)
    )
    mask = ~(nz < -1 + eps)
    tangent = normalize(
        torch.where(mask, torch.cat([1 - nx * nx * a, b, -nx], dim=-1), vecx)
    )
    bitangent = normalize(
        torch.where(mask, torch.cat([b, 1 - ny * ny * a, -ny], dim=-1), vecy)
    )
    # up = torch.where(normal[:, 2:3] < 0.999, z_up, x_up)
    # tangent = normalize(torch.linalg.cross(up, normal))
    # bitangent = normalize(torch.linalg.cross(normal, tangent))
    # B, 3, 3
    row_world_basis = torch.stack([tangent, bitangent, normal], dim=1).reshape(B, 3, 3)
    return row_world_basis


def stretch(dir, ax, ay):
    return normalize(
        torch.stack([ax * dir[..., 0], ay * dir[..., 1], dir[..., 2]], dim=-1)
    )


def unstretch(dir, ax, ay):
    return normalize(
        torch.stack([dir[..., 0] * ax, dir[..., 1] * ay, dir[..., 2]], dim=-1)
    )


class GGXSampler(PseudoRandomSampler):
    def sample(
        self,
        u1,
        u2,
        dir_out,
        normal,
        r1,
        r2,
        ray_mask,
        eps=torch.finfo(torch.float32).eps,
        **kwargs
    ):
        num_samples = ray_mask.shape[1]
        # no non isotropic stuff
        r2 = r1
        # dir_out: (B, 3)
        # normal: (B, 3)
        # r1, r2: B roughness values for anisotropic roughness
        device = normal.device
        B = normal.shape[0]

        # establish basis for BRDF
        z_up = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, 3).expand(B, 3)
        x_up = torch.tensor([-1.0, 0.0, 0.0], device=device).reshape(1, 3).expand(B, 3)
        up = torch.where(normal[:, 2:3].abs() < 0.999, z_up, x_up)
        tangent = normalize(torch.linalg.cross(up, normal))
        bitangent = normalize(torch.linalg.cross(normal, tangent))
        # B, 3, 3
        row_world_basis = torch.stack([tangent, bitangent, normal], dim=1).reshape(
            B, 3, 3
        )
        # row_world_basis = coordinate_frame(normal)

        # GGXVNDF
        # V_l = torch.matmul(torch.inverse(row_world_basis.permute(0, 2, 1)), dir_out.unsqueeze(-1)).squeeze(-1)
        # ic((normal*dir_out).sum(dim=-1).min(), (normal*dir_out).sum(dim=-1).max())
        # ic(1, V_l.min(dim=0), V_l.max(dim=0))
        V_l = torch.matmul(row_world_basis, dir_out.unsqueeze(-1)).squeeze(-1)
        # ic(2, V_l.min(dim=0), V_l.max(dim=0))
        r1_c = r1.squeeze(-1)
        r2_c = r2.squeeze(-1)

        V_stretch = stretch(V_l, r1_c, r2_c).unsqueeze(1)
        # hemi_frame = coordinate_frame(V_stretch)
        # T1 = hemi_frame[:, 0:1]
        # T2 = hemi_frame[:, 1:2]
        # ic(hemi_frame)
        T1 = torch.where(
            V_stretch[..., 2:3] < 0.999,
            normalize(torch.linalg.cross(V_stretch, z_up.unsqueeze(1), dim=-1)),
            x_up.unsqueeze(1),
        )
        T2 = normalize(torch.linalg.cross(T1, V_stretch, dim=-1))

        z = V_stretch[..., 2].reshape(-1, 1)
        a = (1 / (1 + z.detach()).clip(min=1e-8)).clip(max=1e4)

        # stretch and mask stuff to reduce memory
        a_mask = a.expand(u1.shape)[ray_mask]

        r_mask1 = r1_c.reshape(-1, 1).expand(u1.shape)[ray_mask]
        r_mask2 = r2_c.reshape(-1, 1).expand(u1.shape)[ray_mask]

        z_mask = z.expand(u1.shape)[ray_mask]
        u1_mask = u1[ray_mask]
        u2_mask = u2[ray_mask]
        T1_mask = T1.expand(-1, num_samples, 3)[ray_mask]
        T2_mask = T2.expand(-1, num_samples, 3)[ray_mask]
        V_stretch_mask = V_stretch.expand(-1, num_samples, 3)[ray_mask]
        row_world_basis_mask = (
            row_world_basis.permute(0, 2, 1)
            .reshape(B, 1, 3, 3)
            .expand(B, num_samples, 3, 3)[ray_mask]
        )

        r = torch.sqrt(u1_mask)
        phi = torch.where(
            u2_mask < a_mask,
            u2_mask / a_mask * math.pi,
            (u2_mask - a_mask) / (1 - a_mask) * math.pi + math.pi,
        )
        P1 = (r * safemath.safe_cos(phi)).unsqueeze(-1)
        P2 = (
            r
            * safemath.safe_sin(phi)
            * torch.where(u2_mask < a_mask, torch.tensor(1.0, device=device), z_mask)
        ).unsqueeze(-1)
        # ic((1-a).min(), a.min(), a.max(), phi.min(), phi.max(), (1-a).max())
        N_stretch = (
            P1 * T1_mask
            + P2 * T2_mask
            + (1 - P1 * P1 - P2 * P2).clip(min=eps).sqrt() * V_stretch_mask
        )

        H_l = unstretch(N_stretch, r_mask1, r_mask2)

        # H_l = normalize(
        #     torch.stack(
        #         [
        #             r_mask1 * N_stretch[..., 0],
        #             r_mask2 * N_stretch[..., 1],
        #             N_stretch[..., 2].clip(min=0),
        #         ],
        #         dim=-1,
        #     )
        # )
        # H_l = normalize(
        #     torch.stack(
        #         [
        #             r_mask1 * N_stretch[..., 0],
        #             r_mask2 * N_stretch[..., 1],
        #             N_stretch[..., 2],
        #         ],
        #         dim=-1,
        #     )
        # )

        # first = torch.zeros_like(ray_mask)
        # first[:, 0] = True
        # H_l[first[ray_mask], 0] = 0
        # H_l[first[ray_mask], 1] = 0
        # H_l[first[ray_mask], 2] = 1

        H = torch.matmul(row_world_basis_mask, H_l.unsqueeze(-1)).squeeze(-1)
        # H = torch.einsum('bni,bij->bnj', H_l, row_world_basis)

        omega_o = dir_out.unsqueeze(1).expand(-1, num_samples, 3)[ray_mask]
        eN = normal.unsqueeze(1).expand(-1, num_samples, 3)[ray_mask]
        # flip so always visible
        omega_i = normalize(2.0 * (omega_o * H).sum(dim=-1, keepdim=True) * H - omega_o)
        sign = torch.where((omega_i * eN).sum(dim=-1, keepdim=True) > 0, 1, -1)
        omega_i = omega_i * sign

        lomega_i = torch.matmul(
            row_world_basis_mask.permute(0, 2, 1), omega_i.unsqueeze(-1)
        ).squeeze(-1)
        lomega_o = torch.matmul(
            row_world_basis_mask.permute(0, 2, 1), omega_o.unsqueeze(-1)
        ).squeeze(-1)

        # prob = -(math.pi * r_mask1 * r_mask2 * (
        #     H_l[:, 0]**2 / (r_mask1**2).clip(min=eps) +
        #     H_l[:, 1]**2 / (r_mask2**2).clip(min=eps) +
        #     H_l[:, 2]**2
        #     )**2).clip(min=eps)
        # logD = 2 * torch.log(r_mask1.clip(min=eps)) - torch.log(
        #     (math.pi * (H_l[:, 2] ** 2 * (r_mask1**2 - 1) + 1) ** 2).clip(min=eps)
        # )
        # logD = -torch.log(
        #     r_mask1**2
        #     * (
        #         H_l[:, 0] ** 2 / r_mask1**2
        #         + H_l[:, 1] ** 2 / r_mask1**2
        #         + H_l[:, 2] ** 2
        #     )
        #     ** 2
        # )
        with torch.no_grad():
            logD = (
                self.compute_prob(lomega_i, lomega_o, H_l, r_mask1, r_mask2)
                .clip(min=eps)
                .log()
                .reshape(-1)
            )

        return omega_i, row_world_basis_mask, logD

    def compute_prob(self, dir_in, dir_out, halfvec, r1, r2, **kwargs):
        # uses halfvec in the local shading frame
        eps = torch.finfo(torch.float32).eps
        r2 = r1.reshape(-1).clip(min=eps)
        r1 = (r1 + r2).reshape(-1).clip(min=eps) / 2
        # n_dot_in = ldir_in[:, 2]
        # NdotH = halfvec[..., 2].abs().clip(min=eps, max=1).reshape(-1, 1)
        # logD = 2 * torch.log(r1.clip(min=eps)) - torch.log(
        #     (math.pi * (NdotH**2 * (r1**2 - 1) + 1) ** 2).clip(min=eps)
        # )
        n_dot_out = dir_out[..., 2]
        Lambda = (
            -1
            + (
                1
                + ((dir_in[:, 0] * r1) ** 2 + (dir_in[:, 1] * r2) ** 2)
                / (dir_in[:, 2] ** 2).clip(min=1e-6)
            )
            .clip(min=eps)
            .sqrt()
        ) / 2
        invG = 1 + Lambda

        invD = (
            math.pi
            * r1
            * r2
            * (
                halfvec[:, 0] ** 2 / r1**2
                + halfvec[:, 1] ** 2 / r2**2
                + halfvec[:, 2] ** 2
            )
            ** 2
        )
        logD = -(invG * invD).clip(min=eps).log() - (4 * n_dot_out).clip(min=eps).log()
        # ic(1 / invG, 1 / invD, n_dot_out, halfvec, r1, r2)
        # logD = -(4 * n_dot_in).clip(min=eps).log()
        # ic(r1.shape, halfvec.shape, logD.shape)
        prob = logD.exp().reshape(-1, 1)
        masked = torch.where((dir_in[:, 2:3] > 0), prob, torch.zeros_like(prob))
        return masked
