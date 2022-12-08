import torch
import numpy as np

def _create_mlp(input_w, output_w, num_layers, hidden_w=128, bias=True):
    if num_layers == 0:
        net = torch.nn.Sequential(torch.nn.Identity())
    elif num_layers == 1:
        net = torch.nn.Sequential(torch.nn.Linear(input_w, output_w, bias=bias))
    else:
        net = torch.nn.Sequential(
            torch.nn.Linear(input_w, hidden_w),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(hidden_w, hidden_w),
                ] for _ in range(num_layers-2)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_w, output_w, bias=bias),
        )
    return net

class SkipConnection(torch.nn.Module):
    def __init__(self, input_w, output_w, num_layers, hidden_w=128, skip=4, bias=True, **kwargs):
        super().__init__()
        self.mlp1 = _create_mlp(input_w, hidden_w, skip, hidden_w)
        self.mlp2 = _create_mlp(input_w+hidden_w, output_w, num_layers-skip, hidden_w)

    def forward(self, x):
        sx = self.mlp1(x)
        sx = torch.relu(sx)
        x = torch.cat([x, sx], dim=-1)
        x = self.mlp2(x)
        return x

    def apply(self, *args, **kwargs):
        self.mlp1.apply(*args, **kwargs)
        self.mlp2.apply(*args, **kwargs)


def init_weights_kaiming(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def init_weights_kaiming_fanout(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def init_weights_xavier_sigmoid(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def create_mlp(input_w, output_w, num_layers, hidden_w=128, skip=None, initializer=None, bias=True, **kwargs):
    if skip is None:
        net = _create_mlp(input_w, output_w, num_layers, hidden_w, bias)
    else:
        net = SkipConnection(input_w, output_w, num_layers, hidden_w, skip, bias, **kwargs)
    match initializer:
        case 'kaiming':
            net.apply(init_weights_kaiming)
        case 'xavier':
            net.apply(init_weights_xavier)
        case 'xavier_sigmoid':
            net.apply(init_weights_xavier_sigmoid)
        case _:
            pass
    return net
