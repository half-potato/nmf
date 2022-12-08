import torch

class PseudoRandomSampler(torch.nn.Module):
    def __init__(self, max_samples) -> None:
        super().__init__()
        self.sampler = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        self.max_samples = max_samples
        angs = self.sampler.draw(max_samples)
        self.register_buffer('angs', angs)

    def draw(self, B, num_samples):
        if num_samples > self.max_samples:
            self.max_samples = num_samples
            self.angs = self.sampler.draw(self.max_samples)
        angs = self.angs.reshape(1, self.max_samples, 2)[:, :num_samples, :].expand(B, num_samples, 2)
        # self.sampler = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        # add random offset
        offset = torch.rand(B, 1, 2, device=angs.device)*0.25
        angs = (angs + offset) % 1.0
        return angs

    def update(self, *args, **kwargs):
        pass

