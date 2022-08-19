import torch
import numpy as np
from icecream import ic

def arccos(x):
    return torch.arccos(x.clip(min=-1+1e-8, max=1-1e-8))

class safe_atan2(torch.autograd.Function):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return torch.atan2(x, y)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        eps = 1e-5
        dx, dy = None, None
        if ctx.needs_input_grad[0]:
            dx = grad_output * y / (x**2 + y**2+eps)
        if ctx.needs_input_grad[1]:
            dy = grad_output * -x / (x**2 + y**2+eps)
        # print(x**2 + y**2+eps)
        # if max(torch.max(torch.abs(dx)), torch.max(torch.abs(dy))) > 1:
        #     print(x**2 + y**2+eps)
        #     print(x, y)
        return dx, dy

atan2 = safe_atan2.apply

def safe_trig_helper(x, fn, t=100 * np.pi):
  return fn(torch.where(torch.abs(x) < t, x, x % t))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.sin)

def expected_sin(x, x_var):
  """Estimates mean and variance of sin(z), z ~ N(x, var)."""
  # When the variance is wide, shrink sin towards zero.
  y = torch.exp(-0.5 * x_var) * safe_sin(x)
  y_var = 0.5 * (1 - torch.exp(-2 * x_var) * safe_cos(2 * x)) - y**2
  y_var = y_var.clamp(min=0)
  return y, y_var

def integrated_pos_enc(x_coord, min_deg, max_deg, diag=True):
  """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].
  Args:
    x_coord: a tuple containing: (x, x_cov), 
      x: (B, N), variables to be encoded. Should be in [-pi, pi].
      x_cov: (B, N), covariance matrices for `x`.
    min_deg: int, the min degree of the encoding.
    max_deg: int, the max degree of the encoding.
    diag: bool, if true, expects input covariances to be diagonal (full
      otherwise).
  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if diag:
    x, x_cov_diag = x_coord
    device = x.device
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=device)
    shape = list(x.shape[:-1]) + [-1]
    y = torch.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)
  else:
    x, x_cov = x_coord
    device = x.device
    num_dims = x.shape[-1]
    basis = torch.cat(
        [2**i * torch.eye(num_dims, device=device) for i in range(min_deg, max_deg)], 1)
    y = torch.matmul(x, basis)
    # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
    # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
    y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)

  return expected_sin(
      torch.cat([y, y + 0.5 * torch.pi], axis=-1),
      torch.cat([y_var] * 2, axis=-1))[0]
