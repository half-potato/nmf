import torch

class Tonemap(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    def white(self):
        self.tonemap(torch.tensor(1)).item()

class SRGBTonemap(Tonemap):
    def __init__(self):
        super().__init__()

    def forward(self, img, noclip=False):
        # linear to SRGB
        # img from 0 to 1
        limit = 0.0031308
        out = torch.where(img > limit, 1.055 * (img.clip(min=limit) ** (1.0 / 2.4)) - 0.055, 12.92 * img)
        # mask = img > limit
        # out = torch.zeros_like(img)
        # out[mask] = 1.055 * (img[mask] ** (1.0 / 2.4)) - 0.055
        # out[~mask] = 12.92 * img[~mask]
        if not noclip:
            out = out.clip(0, 1)
        return out

    def inverse(self, img):
        # SRGB to linear
        # img from 0 to 1
        limit = 0.04045
        return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

class HDRTonemap(Tonemap):
    def __init__(self):
        super().__init__()

    def forward(self, img, noclip=False):
        # linear to HDR
        # reinhard hdr mapping + gamma correction
        out = (img / (img.clip(min=0)+1))**(1/2.2)
        if not noclip:
            out = out.clip(0, 1)
        return out

    def inverse(self, img):
        # HDR to linear
        img = img ** 2.2
        return - img / (img-1)

class LinearTonemap(Tonemap):
    def __init__(self):
        super().__init__()

    def forward(self, img, noclip=False):
        # linear to HDR
        if not noclip:
            img = img.clip(0, 1)
        return img

    def inverse(self, img):
        # HDR to linear
        return img
