import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import torch
from icecream import ic

class Logger:
    def __init__(self, enable=True, path=None) -> None:
        if path is not None:
            with open(path, 'rb') as f:
                self.data = pkl.load(f)
        else:
            self.reset()
        self.enable = enable

    def reset(self):
        self.data = defaultdict(list)

    def log_rays(self, rays, recur, return_data):
        if not self.enable:
            return
        # rays: (N, 9) torch tensor of rays
        # recur: level of recursion of rendering
        # return_data: dictionary returned from rendering
        N = rays.shape[0]
        # ic(return_data['depth_map'].shape, rays.shape)
        lines = torch.cat([
            rays[:, :3],
            rays[:, :3] + rays[:, 3:6] * return_data['depth_map'].reshape(-1, 1)
        ], dim=-1)
        self.data['lines'].append(lines)
        self.data['recur'].append(torch.tensor(recur).expand(N))

    def plot_rays(self):
        lines = torch.cat(self.data['lines'], dim=0)
        recurs = torch.cat(self.data['recur'], dim=0)
        N = lines.shape[0]
        assert(recurs.shape[0] == N)
        recur_cols = torch.tensor([
            [0, 0, 0],
            [255, 0, 0],
            [255, 255, 0],
            [255, 255, 255],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
        ])[recurs]
        # construct line data
        lines_l = []
        for i in range(N):
            lines_l.append(tuple(lines[i][:3]))
            lines_l.append(tuple(lines[i][3:6]))
            lines_l.append((None, None, None))
        lx, ly, lz = zip(*lines_l)
        go1 = go.Scatter3d(lx, ly, lz, mode='lines')
        fig = go.Figure([go1])
        fig.show()

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self.data, f)
