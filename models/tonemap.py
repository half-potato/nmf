import torch


class SRGBTonemap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img, noclip=False):
        # linear to SRGB
        # img from 0 to 1
        limit = 0.0031308
        img = torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
        if not noclip:
            img = img.clip(0, 1)
        return img

    def inverse(self, img):
        # SRGB to linear
        # img from 0 to 1
        limit = 0.04045
        return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

class HDRTonemap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img):
        # linear to HDR
        return img

    def inverse(self, img):
        # HDR to linear
        return img

class LinearTonemap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img):
        # linear to HDR
        return img.clip(0, 1)

    def inverse(self, img):
        # HDR to linear
        return img
