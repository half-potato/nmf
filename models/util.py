import torch
import numpy as np

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

def init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def create_mlp(input_w, output_w, num_layers, hidden_w=128, initializer=None, bias=True, **kwargs):
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
    match initializer:
        case 'kaiming':
            net.apply(init_weights_kaiming)
        case 'xavier':
            net.apply(init_weights_xavier)
        case _:
            pass
    return net
