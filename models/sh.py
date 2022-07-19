import torch
import math

################## sh function ##################
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]
C5 = [
    math.sqrt(1/2)*3/16*math.sqrt(77/math.pi),
    math.sqrt(1/2)*3/2*math.sqrt(385/2/math.pi),
    math.sqrt(1/2)*1/16*math.sqrt(385/math.pi),
    math.sqrt(1/2)*1/2*math.sqrt(1155/2/math.pi),
    math.sqrt(1/2)*1/8*math.sqrt(165/2/math.pi),
    1/16*math.sqrt(11/math.pi),
    math.sqrt(1/2)*1/8*math.sqrt(165/2/math.pi),
    math.sqrt(1/2)*1/4*math.sqrt(1155/2/math.pi),
    math.sqrt(1/2)*1/16*math.sqrt(385/math.pi),
    math.sqrt(1/2)*3/8*math.sqrt(385/math.pi),
    math.sqrt(1/2)*3/16*math.sqrt(77/math.pi),
]
C6 = [
    math.sqrt(1/2)*1/16*math.sqrt(3003/math.pi),
    math.sqrt(1/2)*3/16*math.sqrt(1001/math.pi),
    math.sqrt(1/2)*3/4*math.sqrt(91/2/math.pi),
    math.sqrt(1/2)*1/16*math.sqrt(1365/math.pi),
    math.sqrt(1/2)*1/16*math.sqrt(1365/math.pi),
    math.sqrt(1/2)*1/8*math.sqrt(273/math.pi),
    1/32*math.sqrt(13/math.pi),
    math.sqrt(1/2)*1/8*math.sqrt(273/math.pi),
    math.sqrt(1/2)*1/16*math.sqrt(1365/math.pi),
    math.sqrt(1/2)*1/16*math.sqrt(1365/math.pi),
    math.sqrt(1/2)*3/16*math.sqrt(91/2/math.pi),
    math.sqrt(1/2)*3/16*math.sqrt(1001/math.pi),
    math.sqrt(1/2)*1/32*math.sqrt(3003/math.pi),
]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    :param deg: int SH max degree. Currently, 0-4 supported
    :param sh: torch.Tensor SH coeffs (..., C, (max degree + 1) ** 2)
    :param dirs: torch.Tensor unit directions (..., 3)
    :return: (..., C)
    """
    assert deg <= 4 and deg >= 0
    assert (deg + 1) ** 2 == sh.shape[-1]
    C = sh.shape[-2]

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                        C3[1] * xy * z * sh[..., 10] +
                        C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                        C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                        C3[5] * z * (xx - yy) * sh[..., 14] +
                        C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def eval_sh_bases(deg, dirs):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param deg: int SH max degree. Currently, 0-4 supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., (deg+1) ** 2)
    """
    assert deg <= 6 and deg >= 0
    result = torch.empty((*dirs.shape[:-1], (deg + 1) ** 2), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y;
        result[..., 2] = C1 * z;
        result[..., 3] = -C1 * x;
        if deg == 1:
            return result
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    x4, y4, z4 = x**4, y**4, z**4
    xy2 = xy**2
    if deg > 1:
        result[..., 4] = C2[0] * xy;
        result[..., 5] = C2[1] * yz;
        result[..., 6] = C2[2] * (2.0 * zz - xx - yy);
        result[..., 7] = C2[3] * xz;
        result[..., 8] = C2[4] * (xx - yy);

    if deg > 2:
        result[..., 9] = C3[0] * y * (3 * xx - yy);
        result[..., 10] = C3[1] * xy * z;
        result[..., 11] = C3[2] * y * (4 * zz - xx - yy);
        result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
        result[..., 13] = C3[4] * x * (4 * zz - xx - yy);
        result[..., 14] = C3[5] * z * (xx - yy);
        result[..., 15] = C3[6] * x * (xx - 3 * yy);

    if deg > 3:
        result[..., 16] = C4[0] * xy * (xx - yy);
        result[..., 17] = C4[1] * yz * (3 * xx - yy);
        result[..., 18] = C4[2] * xy * (7 * zz - 1);
        result[..., 19] = C4[3] * yz * (7 * zz - 3);
        result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3);
        result[..., 21] = C4[5] * xz * (7 * zz - 3);
        result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1);
        result[..., 23] = C4[7] * xz * (xx - 3 * yy);
        result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));

    if deg > 4:
        result[..., 25] = C5[0] * y * (5*x4 - 10*xy2 + y4)
        result[..., 26] = C5[1] * z * (x4 - 6*xy2 + y4)
        result[..., 27] = C5[2] * y * (3*xx-yy)*(9*zz-1)
        result[..., 28] = C5[3] * z * (xx-yy)*(3*zz-1)
        result[..., 29] = C5[4] * y * (21*z4-14*zz+1)
        result[..., 30] = C5[5] * z * (15-70*zz+63*z4)
        result[..., 31] = C5[6] * x * (1-14*zz+21*z4)
        result[..., 32] = C5[7] * x*y*z*(3*zz-1)
        result[..., 33] = C5[8] * x*(xx-3*yy) * (9*zz-1)
        result[..., 34] = C5[9] * x * y * z * (xx-yy)
        result[..., 35] = C5[10] * x * (x4-10*xy2+5*y4)
    if deg <= 5:
        return result

    z6 = z**6

    if deg > 5:
        result[..., 36] = C6[0] * xy*(3*x4-10*xy2+3*y4)
        result[..., 37] = C6[1] * y * (5*x4-10*xy2+y4) * z
        result[..., 38] = C6[2] * xy*(xx-yy)*(11*zz-1)
        result[..., 39] = C6[3] * z*y*(3*xx-yy)*(11*zz-3)
        result[..., 40] = C6[4] * xy*(1-18*zz+33*z4)
        result[..., 41] = C6[6] * x*z*(5-30*zz+33*z4)
        result[..., 42] = C6[5] * (-5+105*zz-315*z4+231*z6)
        result[..., 43] = C6[6] * x*z*(5-30*zz+33*z4)
        result[..., 44] = C6[7] * (xx-yy)*(1-18*zz+33*z4)
        result[..., 45] = C6[8] * x*z*(3*xx-yy)*(11*zz-3)
        result[..., 46] = C6[9] * (x4-6*xy2+y4)*(11*zz-1)
        result[..., 47] = C6[10] * x*(x4-10*xy2+5*y4)*z
        result[..., 48] = C6[11] * xx*(x4-15*xy2)+yy*(15*xy2-y4)
    return result
