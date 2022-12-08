import torch
from mutils import normalize
from .base import PseudoRandomSampler

class CosineLobeSampler(PseudoRandomSampler):
    def sample(self, viewdir, normal, r1, r2, ray_mask):
        num_samples = ray_mask.shape[1]
        # viewdir: (B, 3)
        # normal: (B, 3)
        # r1, r2: B roughness values for anisotropic roughness
        device = normal.device
        B = normal.shape[0]

        # establish basis for BRDF
        z_up = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, 3).expand(B, 3)
        x_up = torch.tensor([-1.0, 0.0, 0.0], device=device).reshape(1, 3).expand(B, 3)
        up = torch.where(normal[:, 2:3] < 0.9, z_up, x_up)
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

        angs = self.draw(B, num_samples).to(device)

        # here is where things get really large
        u1 = angs[..., 0]
        u2 = angs[..., 1]

        # stretch and mask stuff to reduce memory
        r_mask1 = r1_c.reshape(-1, 1).expand(u1.shape)[ray_mask]
        r_mask2 = r2_c.reshape(-1, 1).expand(u1.shape)[ray_mask]

        u1_mask = u1[ray_mask]
        u2_mask = u2[ray_mask]
        row_world_basis_mask = row_world_basis.permute(0, 2, 1).reshape(B, 1, 3, 3).expand(B, num_samples, 3, 3)[ray_mask]

        theta = u1_mask * math.pi
        phi = 2 * u2_mask * math.pi
        sphere_noise = torch.stack([
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            -torch.sin(theta),
        ], dim=-1)

        # so this function is the inverse of the CDF
        H_l = normalize(r_mask1.reshape(-1, 1) * sphere_noise + torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, -1))

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

        return L, row_world_basis_mask

    def calculate_mipval(self, H, V, N, ray_mask, roughness, eps=torch.finfo(torch.float32).eps):
        num_samples = ray_mask.shape[1]
        H_l = torch.matmul(row_world_basis.permute(0, 2, 1), H.unsqueeze(-1)).squeeze(-1)
        sphere_noise = (H_l - torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, -1)) / roughness

        NdotH = (H * N).sum(dim=-1).abs().clip(min=eps, max=1)
        HdotV = (H * V).sum(dim=-1).abs().clip(min=eps, max=1)
        NdotV = (N * V).sum(dim=-1).abs().clip(min=eps, max=1)
        logD = 2*torch.log(roughness.clip(min=eps))# - 2*torch.log((NdotH**2*(roughness**2-1)+1).clip(min=eps))
        # ic(NdotH.shape, NdotH, D, D.mean())
        # px.scatter(x=NdotH[0].detach().cpu().flatten(), y=D[0].detach().cpu().flatten()).show()
        # assert(False)
        lpdf = logD# + torch.log(HdotV) - torch.log(NdotV)
        # pdf = D * HdotV / NdotV / roughness.reshape(-1, 1)
        # pdf = NdotH / 4 / HdotV
        # pdf = D# / NdotH
        indiv_num_samples = ray_mask.sum(dim=1, keepdim=True).expand(-1, num_samples)[ray_mask]
        mipval = -torch.log(indiv_num_samples.clip(min=1)) - lpdf
        return mipval


class Phong(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lr = 0

    def forward(self, incoming_light, V, L, N, features, matprop, mask, ray_mask):
        B, M, _ = L.shape
        # refdir = L[:, 0:1, :]
        refdir = 2*(N * L).sum(dim=-1, keepdim=True) * N - L
        RdotV = (refdir * V.reshape(-1, 1, 3)).sum(dim=-1, keepdim=True).clip(min=1e-8)#.reshape(-1, 1, 1)
        # ic(RdotV, V, refdir)
        LdotN = (L * N).sum(dim=-1, keepdim=True).clip(min=1e-8)
        tint = matprop['tint'][mask].reshape(-1, 1, 3)
        f0 = matprop['f0'][mask].reshape(-1, 1, 3)
        alpha = matprop['reflectivity'][mask].reshape(-1, 1, 1)
        albedo = matprop['albedo'][mask].reshape(-1, 3)
        diffuse = tint * LdotN
        specular = f0 * RdotV**alpha
        output = albedo + ((diffuse + specular) * incoming_light).sum(dim=1) / M

        return output

