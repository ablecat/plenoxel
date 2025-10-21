"""Runtime Plenoxel field sampling implementation."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from .aabb import AABB
from .dda import GridDDA
from .field import IBgField, SampleOpt
from .io_npz import PlenoxelAsset, build_dense_occupancy
from .sh import apply_sh_rgb, eval_sh_basis
from .utils import TransformHelper, compute_voxel_size


class PlenoxelField(IBgField):
    """Sample Plenoxel assets using short-range volume rendering."""

    def __init__(self, asset: PlenoxelAsset, transform: TransformHelper):
        self.asset = asset
        self.transform = transform
        self.aabb = AABB(asset.aabb_min, asset.aabb_max)
        self.voxel_size = compute_voxel_size(asset.aabb_min, asset.aabb_max, asset.res)
        self.occupancy = self._prepare_occupancy(asset)
        self.index_lookup = self._build_lookup(asset.idx_xyz)

    @staticmethod
    def _prepare_occupancy(asset: PlenoxelAsset) -> np.ndarray:
        if asset.occ_pyr is not None:
            occ = asset.occ_pyr[-1]
            return occ.astype(bool)
        return build_dense_occupancy(asset)

    @staticmethod
    def _build_lookup(idx_xyz: np.ndarray) -> Dict[Tuple[int, int, int], int]:
        mapping: Dict[Tuple[int, int, int], int] = {}
        for i, idx in enumerate(idx_xyz):
            mapping[tuple(int(v) for v in idx.tolist())] = i
        return mapping

    def aabb_world(self) -> np.ndarray:
        corners = np.array(
            [
                [self.asset.aabb_min[0], self.asset.aabb_min[1], self.asset.aabb_min[2]],
                [self.asset.aabb_max[0], self.asset.aabb_min[1], self.asset.aabb_min[2]],
                [self.asset.aabb_min[0], self.asset.aabb_max[1], self.asset.aabb_min[2]],
                [self.asset.aabb_min[0], self.asset.aabb_min[1], self.asset.aabb_max[2]],
                [self.asset.aabb_max[0], self.asset.aabb_max[1], self.asset.aabb_min[2]],
                [self.asset.aabb_max[0], self.asset.aabb_min[1], self.asset.aabb_max[2]],
                [self.asset.aabb_min[0], self.asset.aabb_max[1], self.asset.aabb_max[2]],
                [self.asset.aabb_max[0], self.asset.aabb_max[1], self.asset.aabb_max[2]],
            ],
            dtype=np.float32,
        )
        world = np.stack([self.transform.field_to_world_point(c) for c in corners], axis=0)
        return np.stack([np.min(world, axis=0), np.max(world, axis=0)], axis=0)

    def is_ready(self) -> bool:
        return True

    def sample_rgb(self, x_world: np.ndarray, dir_world: np.ndarray, opt: SampleOpt) -> np.ndarray:
        if not opt.linear_rgb:
            raise ValueError("Plenoxel field only outputs linear RGB")
        x_world = np.asarray(x_world, dtype=np.float32)
        dir_world = np.asarray(dir_world, dtype=np.float32)
        dir_norm = float(np.linalg.norm(dir_world))
        if dir_norm == 0.0:
            return np.zeros(3, dtype=np.float32)
        dir_world = dir_world / dir_norm
        origin_world = x_world + dir_world * opt.t_near_bias
        origin_field = self.transform.world_to_field_point(origin_world)
        dir_field = self.transform.world_to_field_dir(dir_world)
        dir_field_norm = np.linalg.norm(dir_field)
        if dir_field_norm == 0.0:
            return np.zeros(3, dtype=np.float32)
        dir_field = dir_field / dir_field_norm

        hit, t0, t1 = self.aabb.intersect(origin_field, dir_field)
        if not hit:
            return np.zeros(3, dtype=np.float32)

        t_far = min(t1, opt.t_far_max)
        t_start = max(t0, 0.0)
        dda = GridDDA(origin_field, dir_field, self.asset.aabb_min, self.voxel_size, self.asset.res, t_start)
        rgb = np.zeros(3, dtype=np.float32)
        trans = 1.0
        steps = 0
        basis_dir = dir_world
        basis = eval_sh_basis(self.asset.sh_deg, basis_dir)

        while steps < opt.max_steps and trans > opt.tau_min and dda.inside_bounds():
            t_before = dda.current_t()
            cell, _, _ = dda.step()
            t_after = dda.current_t()
            if t_before > t_far:
                break
            t_in = max(t_before, t_start)
            t_out = min(t_after, t_far)
            dt_eff = t_out - t_in
            if dt_eff <= 0:
                if t_after > t_far:
                    break
                continue
            if not self.occupancy[tuple(cell)]:
                if t_after > t_far:
                    break
                continue
            idx = self.index_lookup.get(tuple(int(v) for v in cell))
            if idx is None:
                if t_after > t_far:
                    break
                continue
            sigma = float(self.asset.sigma[idx])
            if sigma <= 0.0:
                if t_after > t_far:
                    break
                continue
            coeff = self.asset.sh[idx]
            colour = apply_sh_rgb(coeff, basis)
            alpha = 1.0 - math.exp(-sigma * dt_eff)
            rgb += trans * alpha * colour
            trans *= 1.0 - alpha
            steps += 1
            if t_after > t_far:
                break
        return rgb.astype(np.float32)
