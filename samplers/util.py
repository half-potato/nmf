# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from icecream import ic

def lift_gaussian(d, t_mean, t_var, r_var, diag):
  """Lift a Gaussian defined along a ray to 3D coordinates."""
  mean = d[..., None, :] * t_mean[..., None]
  device = d.device

  d_mag_sq = torch.sum(d**2, dim=-1, keepdim=True).clip(min=1e-10)

  if diag:
    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
  else:
    d_outer = d[..., :, None] * d[..., None, :]
    eye = torch.eye(d.shape[-1], device=device)
    null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
    t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
    xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
    cov = t_cov + xy_cov
    return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
  """Approximate a conical frustum as a Gaussian distribution (mean+cov).
  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.
  Args:
    d: torch.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).
  Returns:
    a Gaussian (mean and covariance).
  """
  if stable:
    # Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
    mu = (t0 + t1) / 2  # The average of the two `t` values.
    hw = (t1 - t0) / 2  # The half-width of the two `t` values.
    eps = torch.finfo(torch.float32).eps
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2).clip(min=eps)
    denom = (3 * mu**2 + hw**2).clip(min=eps)
    t_var = (hw**2) / 3 - (4 / 15) * hw**4 * (12 * mu**2 - hw**2) / denom**2
    r_var = (mu**2) / 4 + (5 / 12) * hw**2 - (4 / 15) * (hw**4) / denom
  else:
    # Equations 37-39 in the paper.
    t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
    r_var = 3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_var = t_mosq - t_mean**2
  r_var *= base_radius**2
  return lift_gaussian(d, t_mean, t_var, r_var, diag)
