import torch

def grid_sample(input, grid, mode, padding_mode, align_corners):
    return _GridSampleForward.apply(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

class _GridSampleForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, mode, padding_mode, align_corners):
        assert input.ndim == 4
        assert grid.ndim == 4

        if mode == "bilinear":
            mode_enum = 0
        elif mode == "nearest":
            mode_enum = 1
        else:  # mode == 'bicubic'
            mode_enum = 2

        if padding_mode == "zeros":
            padding_mode_enum = 0
        elif padding_mode == "border":
            padding_mode_enum = 1
        else:  # padding_mode == 'reflection'
            padding_mode_enum = 2

        output = torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grid, mode_enum, padding_mode_enum, align_corners = ctx.saved_tensors
        grad_input, grad_grid = _GridSampleBackward.apply(grad_output, input, grid, mode_enum, padding_mode_enum, align_corners)
        return grad_input, grad_grid, None, None, None

class _GridSampleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid, mode_enum, padding_mode_enum, align_corners):
        if grid.shape[-1] == 2:
            op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
        if grid.shape[-1] == 1:
            op = torch._C._jit_get_operation('aten::grid_sampler_1d_backward')
        if grid.shape[-1] == 3:
            op = torch._C._jit_get_operation('aten::grid_sampler_3d_backward')

        grad_input, grad_grid = op(grad_output, input, grid, mode_enum, padding_mode_enum, align_corners, (ctx.needs_input_grad[1], ctx.needs_input_grad[2]))
        ctx.save_for_backward(grid, mode_enum, padding_mode_enum, align_corners)
        return grad_input, grad_grid, None, None, None

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        _ = grad2_grad_grid # unused
        grid, mode_enum, padding_mode_enum, align_corners = ctx.saved_tensors
        grad2_grad_output = None
        grad2_input = None
        grad2_grid = None

        if ctx.needs_input_grad[0]:
            grad2_grad_output = _GridSampleForward.apply(grad2_grad_input, grid, mode_enum, padding_mode_enum, align_corners)

        assert not ctx.needs_input_grad[2]
        return grad2_grad_output, grad2_input, grad2_grid, None, None, None