import torch

def normalize(v, ord=2):
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True, ord=ord)+1e-8)

def expand_bits(v):                                                                                                                                                                                                
    v = (v * 0x00010001) & 0xFF0000FF                                                                                                                                                                              
    v = (v * 0x00000101) & 0x0F00F00F                                                                                                                                                                              
    v = (v * 0x00000011) & 0xC30C30C3                                                                                                                                                                              
    v = (v * 0x00000005) & 0x49249249                                                                                                                                                                              
    return v                                                                                                                                                                                                       

def morton3D(xyz):                                                                                                                                                                                                 
    exyz = expand_bits(xyz)                                                                                                                                                                                        
    return (exyz[..., 0] | (exyz[..., 1] << 1) | (exyz[..., 2] << 2)).long()

