"""Spherical harmonics evaluation for numpy arrays."""
from __future__ import annotations

import math
import numpy as np

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
SH_C4 = (
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
)


def eval_sh_basis(deg: int, direction: np.ndarray) -> np.ndarray:
    """Evaluate real spherical harmonics basis up to degree ``deg``."""

    direction = np.asarray(direction, dtype=np.float32)
    if direction.shape != (3,):
        raise ValueError("direction must be 3-vector")
    basis_dim = (deg + 1) ** 2
    result = np.empty((basis_dim,), dtype=np.float32)
    x, y, z = direction
    result[0] = SH_C0
    if basis_dim == 1:
        return result
    result[1] = -SH_C1 * y
    result[2] = SH_C1 * z
    result[3] = -SH_C1 * x
    if basis_dim <= 4:
        return result
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result[4] = SH_C2[0] * xy
    result[5] = SH_C2[1] * yz
    result[6] = SH_C2[2] * (2.0 * zz - xx - yy)
    result[7] = SH_C2[3] * xz
    result[8] = SH_C2[4] * (xx - yy)
    if basis_dim <= 9:
        return result
    result[9] = SH_C3[0] * y * (3 * xx - yy)
    result[10] = SH_C3[1] * xy * z
    result[11] = SH_C3[2] * y * (4 * zz - xx - yy)
    result[12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    result[13] = SH_C3[4] * x * (4 * zz - xx - yy)
    result[14] = SH_C3[5] * z * (xx - yy)
    result[15] = SH_C3[6] * x * (xx - 3 * yy)
    if basis_dim <= 16:
        return result
    result[16] = SH_C4[0] * xy * (xx - yy)
    result[17] = SH_C4[1] * yz * (3 * xx - yy)
    result[18] = SH_C4[2] * xy * (7 * zz - 1)
    result[19] = SH_C4[3] * yz * (7 * zz - 3)
    result[20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
    result[21] = SH_C4[5] * xz * (7 * zz - 3)
    result[22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
    result[23] = SH_C4[7] * xz * (xx - 3 * yy)
    result[24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    return result


def apply_sh_rgb(coeff: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Apply SH coefficients to basis to obtain RGB colour."""

    coeff = np.asarray(coeff, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    if coeff.shape[-2] != basis.shape[-1]:
        raise ValueError("Coefficient basis mismatch")
    return coeff @ basis
