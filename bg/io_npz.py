"""NPZ asset IO helpers for Plenoxel background fields."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlenoxelAsset:
    idx_xyz: np.ndarray
    sigma: np.ndarray
    sh: np.ndarray
    res: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    sh_deg: int
    sigma_scale: Optional[float] = None
    sh_scale: Optional[float] = None
    occ_pyr: Optional[np.ndarray] = None

    @property
    def basis_dim(self) -> int:
        return self.sh.shape[1]


def _maybe_dequantise(data: np.ndarray, scale: Optional[np.ndarray]) -> np.ndarray:
    if scale is None:
        return data.astype(np.float32)
    return data.astype(np.float32) * float(scale)


def load_npz_asset(path: str) -> PlenoxelAsset:
    """Load a plenoxel asset from NPZ file."""

    with np.load(path) as npz:
        idx_xyz = npz["idx_xyz"].astype(np.int32)
        sigma = npz["sigma"]
        sigma_scale = npz["sigma_scale"] if "sigma_scale" in npz else None
        sh = npz["sh"]
        sh_scale = npz["sh_scale"] if "sh_scale" in npz else None
        res = npz["res"].astype(np.int32)
        aabb_min = npz["aabb_min"].astype(np.float32)
        aabb_max = npz["aabb_max"].astype(np.float32)
        sh_deg = int(npz["sh_deg"]) if "sh_deg" in npz else int(round(np.sqrt(sh.shape[1]) - 1))
        occ_pyr = npz["occ_pyr"].astype(np.uint8) if "occ_pyr" in npz else None

    sigma = _maybe_dequantise(sigma, sigma_scale)
    sh = _maybe_dequantise(sh, sh_scale)

    return PlenoxelAsset(
        idx_xyz=idx_xyz,
        sigma=sigma,
        sh=sh,
        res=res,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        sh_deg=sh_deg,
        sigma_scale=float(sigma_scale) if sigma_scale is not None else None,
        sh_scale=float(sh_scale) if sh_scale is not None else None,
        occ_pyr=occ_pyr,
    )


def build_dense_occupancy(asset: PlenoxelAsset) -> np.ndarray:
    """Construct a dense boolean occupancy grid from sparse indices."""

    occ = np.zeros(asset.res, dtype=bool)
    if asset.idx_xyz.size == 0:
        return occ
    occ[tuple(asset.idx_xyz.T)] = True
    return occ
