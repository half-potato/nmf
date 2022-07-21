import torch
import math
from math import pi, sqrt
from icecream import ic

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

def Al(l, kappa):
    return torch.exp(-l*(l+1)/2/(kappa+1e-8))

def eval_sh_bases_scaled(deg, dirs, kappa):
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
    kappa = kappa.reshape(*dirs.shape[:-1])
    result[..., 0] = Al(0, kappa) * C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        scale1 = Al(1, kappa)
        result[..., 1] = -scale1 * C1 * y;
        result[..., 2] = scale1 * C1 * z;
        result[..., 3] = -scale1 * C1 * x;
        if deg == 1:
            return result
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    x4, y4, z4 = x**4, y**4, z**4
    xy2 = xy**2
    if deg > 1:
        scale2 = Al(2, kappa)
        result[..., 4] = scale2 * C2[0] * xy;
        result[..., 5] = scale2 * C2[1] * yz;
        result[..., 6] = scale2 * C2[2] * (2.0 * zz - xx - yy);
        result[..., 7] = scale2 * C2[3] * xz;
        result[..., 8] = scale2 * C2[4] * (xx - yy);

    if deg > 2:
        scale3 = Al(3, kappa)
        result[..., 9] = scale3 * C3[0] * y * (3 * xx - yy);
        result[..., 10] = scale3 * C3[1] * xy * z;
        result[..., 11] = scale3 * C3[2] * y * (4 * zz - xx - yy);
        result[..., 12] = scale3 * C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
        result[..., 13] = scale3 * C3[4] * x * (4 * zz - xx - yy);
        result[..., 14] = scale3 * C3[5] * z * (xx - yy);
        result[..., 15] = scale3 * C3[6] * x * (xx - 3 * yy);

    if deg > 3:
        scale4 = Al(4, kappa)
        result[..., 16] = scale4 * C4[0] * xy * (xx - yy);
        result[..., 17] = scale4 * C4[1] * yz * (3 * xx - yy);
        result[..., 18] = scale4 * C4[2] * xy * (7 * zz - 1);
        result[..., 19] = scale4 * C4[3] * yz * (7 * zz - 3);
        result[..., 20] = scale4 * C4[4] * (zz * (35 * zz - 30) + 3);
        result[..., 21] = scale4 * C4[5] * xz * (7 * zz - 3);
        result[..., 22] = scale4 * C4[6] * (xx - yy) * (7 * zz - 1);
        result[..., 23] = scale4 * C4[7] * xz * (xx - 3 * yy);
        result[..., 24] = scale4 * C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));

    if deg > 4:
        scale5 = Al(5, kappa)
        result[..., 25] = scale5 * C5[0] * y * (5*x4 - 10*xy2 + y4)
        result[..., 26] = scale5 * C5[1] * z * (x4 - 6*xy2 + y4)
        result[..., 27] = scale5 * C5[2] * y * (3*xx-yy)*(9*zz-1)
        result[..., 28] = scale5 * C5[3] * z * (xx-yy)*(3*zz-1)
        result[..., 29] = scale5 * C5[4] * y * (21*z4-14*zz+1)
        result[..., 30] = scale5 * C5[5] * z * (15-70*zz+63*z4)
        result[..., 31] = scale5 * C5[6] * x * (1-14*zz+21*z4)
        result[..., 32] = scale5 * C5[7] * x*y*z*(3*zz-1)
        result[..., 33] = scale5 * C5[8] * x*(xx-3*yy) * (9*zz-1)
        result[..., 34] = scale5 * C5[9] * x * y * z * (xx-yy)
        result[..., 35] = scale5 * C5[10] * x * (x4-10*xy2+5*y4)
    if deg <= 5:
        return result

    z6 = z**6

    if deg > 5:
        scale6 = Al(6, kappa)
        result[..., 36] = scale6 * C6[0] * xy*(3*x4-10*xy2+3*y4)
        result[..., 37] = scale6 * C6[1] * y * (5*x4-10*xy2+y4) * z
        result[..., 38] = scale6 * C6[2] * xy*(xx-yy)*(11*zz-1)
        result[..., 39] = scale6 * C6[3] * z*y*(3*xx-yy)*(11*zz-3)
        result[..., 40] = scale6 * C6[4] * xy*(1-18*zz+33*z4)
        result[..., 41] = scale6 * C6[6] * x*z*(5-30*zz+33*z4)
        result[..., 42] = scale6 * C6[5] * (-5+105*zz-315*z4+231*z6)
        result[..., 43] = scale6 * C6[6] * x*z*(5-30*zz+33*z4)
        result[..., 44] = scale6 * C6[7] * (xx-yy)*(1-18*zz+33*z4)
        result[..., 45] = scale6 * C6[8] * x*z*(3*xx-yy)*(11*zz-3)
        result[..., 46] = scale6 * C6[9] * (x4-6*xy2+y4)*(11*zz-1)
        result[..., 47] = scale6 * C6[10] * x*(x4-10*xy2+5*y4)*z
        result[..., 48] = scale6 * C6[11] * xx*(x4-15*xy2)+yy*(15*xy2-y4)
    return result


def sh_basis(degs, dirs, kappa):
    # evaluate a single degree
    kappa = kappa.reshape(*dirs.shape[:-1])

    x, y, z = dirs.T
    xx, yy, zz = x * x, y * y, z * z
    x4, y4, z4 = x**4, y**4, z**4
    x6, y6, z6 = x**6, y**6, z**6
    x8, y8, z8 = x**8, y**8, z**8
    x10, y10, z10 = x**10, y**10, z**10
    x12, y12, z12 = x**12, y**12, z**12
    x14, y14, z14 = x**14, y**14, z**14
    x16, y16, z16 = x**16, y**16, z**16

    values = []
    for deg in degs:
        scale = Al(deg, kappa)
        if deg == 0:
            values.append(scale*C0)

        if deg == 1:
            values.extend([
                scale*-0.690988*y,
                scale*0.488603*z,
                scale*-0.345494*x
            ])

        if deg == 2:
            values.extend([
                scale*1.5451*x*y,
                scale*-1.5451*y*z,
                scale*(0.946176*zz - 0.315392),
                scale*-0.772548*x*z,
                scale*(0.386274*xx - 0.386274*yy)
            ])

        if deg == 4:
            values.extend([
                scale*3.54026*x*y*(xx - yy),
                scale*2.50334*y*z*(-3*xx + yy),
                scale*1.33809*x*y*(7*zz - 1),
                scale*0.946175*y*z*(3 - 7*zz),
                scale*(3.70251*z4 - 3.17358*zz + 0.317358),
                scale*0.473087*x*z*(3 - 7*zz),
                scale*(0.334523*xx - 0.334523*yy)*(7*zz - 1),
                scale*-1.25167*x*z*(xx - 3*yy),
                scale*(0.442533*x4 - 2.655198*xx*yy + 0.442533*y4)
            ])

        if deg == 8:
            values.extend([
                scale*8.24686*x*y*(x6 - 7*x4*yy + 7*xx*y4 - y6),
                scale*4.12343*y*z*(-7*x6 + 35*x4*yy - 21*xx*y4 + y6),
                scale*1.50566*x*y*(15*zz - 1)*(3*x4 - 10*xx*yy + 3*y4),
                scale*-4.87891*y*z*(5*zz - 1)*(5*x4 - 10*xx*yy + y4),
                scale*2.70633*x*y*(xx - yy)*(65*z4 - 26*zz + 1),
                scale*1.74693*y*z*(-3*xx + yy)*(39*z4 - 26*zz + 3),
                scale*1.29019*x*y*(143*z6 - 143*z4 + 33*zz - 1),
                scale*-0.154208*y*z*(715*z6 - 1001*z4 + 385*zz - 35),
                scale*(58.47336495*z8 - 109.15028124*z6 + 62.9713161*z4 - 11.4493302*zz + 0.31803695),
                scale*-0.0771038*x*z*(715*z6 - 1001*z4 + 385*zz - 35),
                scale*(0.322548*xx - 0.322548*yy)*(143*z6 - 143*z4 + 33*zz - 1),
                scale*-0.873465*x*z*(xx - 3*yy)*(39*z4 - 26*zz + 3),
                scale*(0.338292*x4 - 2.029752*xx*yy + 0.338292*y4)*(65*z4 - 26*zz + 1),
                scale*-2.43946*x*z*(5*zz - 1)*(x4 - 10*xx*yy + 5*y4),
                scale*(15*zz - 1)*(0.376416*x6 - 5.64624*x4*yy + 5.64624*xx*y4 - 0.376416*y6),
                scale*-2.06172*x*z*(x6 - 21*x4*yy + 35*xx*y4 - 7*y6),
                scale*(0.515429*x8 - 14.432012*x6*yy + 36.08003*x4*y4 - 14.432012*xx*y6 + 0.515429*y8)
            ])

        if deg == 16:
            values.extend([
                scale*19.3994*x*y*(x14 - 35*x12*yy + 273*x10*y4 - 715*x8*y6 + 715*x6*y8 - 273*x4*y10 + 35*xx*y12 - y14),
                scale*6.85872*y*z*(-15*x14 + 455*x12*yy - 3003*x10*y4 + 6435*x8*y6 - 5005*x6*y8 + 1365*x4*y10 - 105*xx*y12 + y14),
                scale*1.74212*x*y*(31*zz - 1)*(7*x12 - 182*x10*yy + 1001*x8*y4 - 1716*x6*y6 + 1001*x4*y8 - 182*xx*y10 + 7*y12),
                scale*-2.75453*y*z*(31*zz - 3)*(13*x12 - 286*x10*yy + 1287*x8*y4 - 1716*x6*y6 + 715*x4*y8 - 78*xx*y10 + y12),
                scale*1.02301*x*y*(899*z4 - 174*zz + 3)*(3*x10 - 55*x8*yy + 198*x6*y4 - 198*x4*y6 + 55*xx*y8 - 3*y10),
                scale*0.605219*y*z*(899*z4 - 290*zz + 15)*(-11*x10 + 165*x8*yy - 462*x6*y4 + 330*x4*y6 - 55*xx*y8 + y10),
                scale*0.285303*x*y*(8091*z6 - 3915*z4 + 405*zz - 5)*(5*x8 - 60*x6*yy + 126*x4*y4 - 60*xx*y6 + 5*y8),
                scale*-0.274925*y*z*(8091*z6 - 5481*z4 + 945*zz - 35)*(9*x8 - 84*x6*yy + 126*x4*y4 - 36*xx*y6 + y8),
                scale*0.777605*x*y*(x6 - 7*x4*yy + 7*xx*y4 - y6)*(40455*z8 - 36540*z6 + 9450*z4 - 700*zz + 7),
                scale*0.476184*y*z*(-7*x6 + 35*x4*yy - 21*xx*y4 + y6)*(13485*z8 - 15660*z6 + 5670*z4 - 700*zz + 21),
                scale*0.0627973*x*y*(3*x4 - 10*xx*yy + 3*y4)*(310155*z10 - 450225*z8 + 217350*z6 - 40250*z4 + 2415*zz - 21),
                scale*-0.0444044*y*z*(5*x4 - 10*xx*yy + y4)*(310155*z10 - 550275*z8 + 341550*z6 - 88550*z4 + 8855*zz - 231),
                scale*0.234966*x*y*(xx - yy)*(310155*z12 - 660330*z10 + 512325*z8 - 177100*z6 + 26565*z4 - 1386*zz + 11),
                scale*0.0728598*y*z*(-3*xx + yy)*(310155*z12 - 780390*z10 + 740025*z8 - 328900*z6 + 69069*z4 - 6006*zz + 143),
                scale*0.00893464*x*y*(5892950.0*z14 - 17298600.0*z12 + 19684700.0*z10 - 10935900.0*z8 + 3062060.0*z6 - 399399*z4 + 19019*zz - 143),
                scale*0.00163124*y*z*(-17678800.0*z14 + 59879900.0*z12 - 80528200.0*z10 + 54679600.0*z8 - 19684700.0*z6 + 3594590.0*z4 - 285285*zz + 6435),
                scale*(14862.935214*z16 - 57533.910858*z14 + 90269.063271*z12 - 73552.588389*z10 + 33098.5905939*z8 - 8058.7928655*z6 + 959.37986754*z4 - 43.280250156*zz + 0.3182371335),
                scale*0.000815618*x*z*(-17678800.0*z14 + 59879900.0*z12 - 80528200.0*z10 + 54679600.0*z8 - 19684700.0*z6 + 3594590.0*z4 - 285285*zz + 6435),
                scale*(0.00223366*xx - 0.00223366*yy)*(5892950.0*z14 - 17298600.0*z12 + 19684700.0*z10 - 10935900.0*z8 + 3062060.0*z6 - 399399*z4 + 19019*zz - 143),
                scale*-0.0364299*x*z*(xx - 3*yy)*(310155*z12 - 780390*z10 + 740025*z8 - 328900*z6 + 69069*z4 - 6006*zz + 143),
                scale*(0.0293707*x4 - 0.1762242*xx*yy + 0.0293707*y4)*(310155*z12 - 660330*z10 + 512325*z8 - 177100*z6 + 26565*z4 - 1386*zz + 11),
                scale*-0.0222022*x*z*(x4 - 10*xx*yy + 5*y4)*(310155*z10 - 550275*z8 + 341550*z6 - 88550*z4 + 8855*zz - 231),
                scale*(0.0156993*x6 - 0.2354895*x4*yy + 0.2354895*xx*y4 - 0.0156993*y6)*(310155*z10 - 450225*z8 + 217350*z6 - 40250*z4 + 2415*zz - 21),
                scale*-0.238092*x*z*(x6 - 21*x4*yy + 35*xx*y4 - 7*y6)*(13485*z8 - 15660*z6 + 5670*z4 - 700*zz + 21),
                scale*(0.0486003*x8 - 1.3608084*x6*yy + 3.402021*x4*y4 - 1.3608084*xx*y6 + 0.0486003*y8)*(40455*z8 - 36540*z6 + 9450*z4 - 700*zz + 7),
                scale*-0.137462*x*z*(8091*z6 - 5481*z4 + 945*zz - 35)*(x8 - 36*x6*yy + 126*x4*y4 - 84*xx*y6 + 9*y8),
                scale*(8091*z6 - 3915*z4 + 405*zz - 5)*(0.0713257*x10 - 3.2096565*x8*yy + 14.978397*x6*y4 - 14.978397*x4*y6 + 3.2096565*xx*y8 - 0.0713257*y10),
                scale*-0.302609*x*z*(899*z4 - 290*zz + 15)*(x10 - 55*x8*yy + 330*x6*y4 - 462*x4*y6 + 165*xx*y8 - 11*y10),
                scale*(899*z4 - 174*zz + 3)*(0.127876*x12 - 8.439816*x10*yy + 63.29862*x8*y4 - 118.157424*x6*y6 + 63.29862*x4*y8 - 8.439816*xx*y10 + 0.127876*y12),
                scale*-1.37727*x*z*(31*zz - 3)*(x12 - 78*x10*yy + 715*x8*y4 - 1716*x6*y6 + 1287*x4*y8 - 286*xx*y10 + 13*y12),
                scale*(31*zz - 1)*(0.435529*x14 - 39.633139*x12*yy + 435.964529*x10*y4 - 1307.893587*x8*y6 + 1307.893587*x6*y8 - 435.964529*x4*y10 + 39.633139*xx*y12 - 0.435529*y14),
                scale*-3.42936*x*z*(x14 - 105*x12*yy + 1365*x10*y4 - 5005*x8*y6 + 6435*x6*y8 - 3003*x4*y10 + 455*xx*y12 - 15*y14),
                scale*(0.606231*x16 - 72.74772*x14*yy + 1103.34042*x12*y4 - 4854.697848*x10*y6 + 7802.19297*x8*y8 - 4854.697848*x6*y10 + 1103.34042*x4*y12 - 72.74772*xx*y14 + 0.606231*y16)
            ])
    return torch.stack(values, dim=-1)
