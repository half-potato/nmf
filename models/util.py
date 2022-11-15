import torch

def create_mlp(input_w, output_w, num_layers, hidden_w=128, initializer=None):
    if num_layers == 0:
        net = torch.nn.Identity()
    elif num_layers == 1:
        net = torch.nn.Linear(input_w, output_w)
    else:
        net = torch.nn.Sequential(
            torch.nn.Linear(input_w, hidden_w),
            *sum([[
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(hidden_w, hidden_w),
                ] for _ in range(num_layers-2)], []),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_w, output_w),
        )
    return net
