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
    
    def log_norms_n_rays(self, xyz, normals, weights):
        if not self.enable:
            return
        self.data['xyz'].append(xyz.reshape(-1, xyz.shape[-1])[..., :3])
        self.data['normals'].append(normals.reshape(-1, 3))
        self.data['weights'].append(weights.reshape(-1, 1))

    def plot_norms(self):
        xyz = torch.cat(self.data['xyz'], dim=0)
        normals = torch.cat(self.data['normals'], dim=0)
        weights = torch.cat(self.data['weights'], dim=0)
        # construct normal data
        N = normals.shape[0]
        lines_l = []
        colors = []
        for i in range(N):
            lines_l.append(tuple(xyz[i][:3]))
            lines_l.append(tuple(xyz[i][:3]+normals[i][3:6]))
            lines_l.append((None, None, None))
            c = int(weights[i]*255)
            colors.extend([f'rgb({c},{c},{c})'] * 3)
        lx, ly, lz = zip(*lines_l)

        go1 = go.Scatter3d(lx, ly, lz, mode='lines')
        fig = go.Figure([go1])
        fig.show()

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
