import torch
import torch.nn.functional as F
from icecream import ic

# 1 = up
# 2 = right
# 3 = down
# 4 = left

class CubeEdge:
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    __slots__ = ['face1', 'edge1', 'face2', 'edge2', 'reverse_depth', 'reverse_edge']
    def __init__(self, face1, edge1, face2, edge2, reverse_depth=False, reverse_edge=False):
        self.face1 = face1
        self.edge1 = edge1
        self.face2 = face2
        self.edge2 = edge2
        self.reverse_depth = reverse_depth
        self.reverse_edge = reverse_edge
# CUBE_EDGES = [
#     CubeEdge(0, CubeEdge.LEFT, 3, CubeEdge.RIGHT),
#     CubeEdge(0, CubeEdge.DOWN, 5, CubeEdge.LEFT),
#     CubeEdge(0, CubeEdge.RIGHT, 2, CubeEdge.LEFT),
#     CubeEdge(0, CubeEdge.UP, 4, CubeEdge.LEFT),
#
#     # face 1 is backwards
#     CubeEdge(1, CubeEdge.LEFT, 3, CubeEdge.LEFT),
#     CubeEdge(1, CubeEdge.DOWN, 5, CubeEdge.RIGHT, reverse_depth=True),
#     CubeEdge(1, CubeEdge.RIGHT, 2, CubeEdge.RIGHT),
#     CubeEdge(1, CubeEdge.UP, 4, CubeEdge.RIGHT, reverse_depth=True),
#
#     # 4 UP, LEFT are taken
#     CubeEdge(2, CubeEdge.UP, 4, CubeEdge.UP),
#     CubeEdge(2, CubeEdge.DOWN, 5, CubeEdge.UP),
#
#     CubeEdge(3, CubeEdge.UP, 4, CubeEdge.DOWN),
#     CubeEdge(3, CubeEdge.DOWN, 5, CubeEdge.DOWN),
# ]

CUBE_EDGES = [
    CubeEdge(0, CubeEdge.LEFT, 4, CubeEdge.RIGHT, reverse_edge=True),
    CubeEdge(0, CubeEdge.DOWN, 2, CubeEdge.RIGHT), # done
    CubeEdge(0, CubeEdge.RIGHT, 5, CubeEdge.LEFT, reverse_edge=True), # done
    CubeEdge(0, CubeEdge.UP, 3, CubeEdge.RIGHT, reverse_depth=True, reverse_edge=True), # done

    # face 1 is backwards
    CubeEdge(1, CubeEdge.LEFT, 5, CubeEdge.RIGHT), # done
    CubeEdge(1, CubeEdge.DOWN, 2, CubeEdge.LEFT, reverse_depth=True, reverse_edge=True), # done
    CubeEdge(1, CubeEdge.RIGHT, 4, CubeEdge.LEFT, reverse_edge=True), # done
    CubeEdge(1, CubeEdge.UP, 3, CubeEdge.LEFT), # done

    # 4 UP, LEFT are taken
    CubeEdge(2, CubeEdge.UP, 4, CubeEdge.DOWN),
    CubeEdge(2, CubeEdge.DOWN, 5, CubeEdge.DOWN, reverse_depth=True),

    CubeEdge(3, CubeEdge.UP, 5, CubeEdge.UP, reverse_depth=True), # done
    CubeEdge(3, CubeEdge.DOWN, 4, CubeEdge.UP, reverse_edge=True), # done
]

def select_edge(img, direction, s, reverse_depth=False, reverse_edge=False, trim_opposite=False):
    if direction == CubeEdge.UP:
        buf = img[..., -s:, :]
    elif direction == CubeEdge.RIGHT:
        buf = img[..., -s:]
    elif direction == CubeEdge.DOWN:
        buf = img[..., :s, :]
    else:
        buf = img[..., :s]
    if trim_opposite:
        if direction == CubeEdge.DOWN or direction == CubeEdge.UP:
            buf = buf[..., s:-s]
        else:
            buf = buf[..., s:-s, :]
    if direction == CubeEdge.DOWN or direction == CubeEdge.UP:
        buf = buf.permute(0, 2, 1)
    if trim_opposite:
        dims = []
        if not reverse_edge:
            dims += [1]
        if reverse_depth:
            dims += [2]
        buf = torch.flip(buf, dims=dims)
    return buf

def cubemap_convolve(input, weight, **kwargs):
    # input: (6, 3, H, W)
    # weight: (O, 3, s, s)
    s = weight.shape[-1]
    coutput = F.conv2d(input, weight, padding=[s, s], **kwargs).float()
    p = s // 2 + 1
    output = coutput[:, :, p:-p, p:-p]
    # return output
    # now, go through and add the border around each output in the correct place
    for edge in CUBE_EDGES:
        out_sel = select_edge(output[edge.face1], edge.edge1, p)
        in_sel = select_edge(coutput[edge.face2], edge.edge2, p, edge.reverse_depth, edge.reverse_edge, trim_opposite=True)
        out_sel += in_sel

        out_sel = select_edge(output[edge.face2], edge.edge2, p)
        in_sel = select_edge(coutput[edge.face1], edge.edge1, p, edge.reverse_depth, edge.reverse_edge, trim_opposite=True)
        out_sel += in_sel
        # break
    return output

def gaussian_fn(M, std, **kwargs):
    n = torch.arange(0, M, **kwargs) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128, **kwargs):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std, **kwargs) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

def create_blur_pyramid(max_size, stds):
    l = len(stds)
    kernel = torch.zeros((3*l, 3, max_size, max_size))
    for i, std in enumerate(stds):
        gkernel = gkern(max_size, std)
        gkernel = gkernel / gkernel.sum()
        kernel[3*i+0, 0] = gkernel
        kernel[3*i+1, 1] = gkernel
        kernel[3*i+2, 2] = gkernel
    return kernel
