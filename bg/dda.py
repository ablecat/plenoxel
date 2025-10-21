"""3D Digital Differential Analyser for sparse grids."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DDAState:
    cell: np.ndarray
    t: float
    t_max: np.ndarray
    t_delta: np.ndarray
    step: np.ndarray


class GridDDA:
    """Iterator stepping through voxels intersected by a ray."""

    def __init__(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        aabb_min: np.ndarray,
        voxel_size: np.ndarray,
        res: np.ndarray,
        t: float,
    ):
        self.origin = origin.astype(np.float32)
        self.direction = direction.astype(np.float32)
        self.aabb_min = aabb_min.astype(np.float32)
        self.voxel_size = voxel_size.astype(np.float32)
        self.res = res.astype(np.int32)
        self.state = self._initialise_state(t)

    def _initialise_state(self, t: float) -> DDAState:
        point = self.origin + self.direction * t
        cell = np.floor((point - self.aabb_min) / self.voxel_size).astype(np.int32)
        cell = np.clip(cell, 0, self.res - 1)
        step = np.sign(self.direction).astype(np.int32)
        step[step == 0] = 1
        next_boundary = self.aabb_min + (cell + (step > 0)) * self.voxel_size
        with np.errstate(divide="ignore"):
            t_max = np.where(
                self.direction != 0,
                (next_boundary - point) / self.direction,
                np.inf,
            )
            t_delta = np.where(
                self.direction != 0,
                self.voxel_size / np.abs(self.direction),
                np.inf,
            )
        t_max = np.maximum(t_max, 0.0)
        return DDAState(cell=cell, t=float(t), t_max=t_max.astype(np.float32), t_delta=t_delta.astype(np.float32), step=step)

    def step(self) -> Tuple[np.ndarray, float, int]:
        axis = int(np.argmin(self.state.t_max))
        t_next = float(self.state.t_max[axis])
        dt = t_next - self.state.t
        cell = self.state.cell.copy()
        self.state.cell[axis] += self.state.step[axis]
        self.state.t = t_next
        self.state.t_max[axis] += self.state.t_delta[axis]
        return cell, dt, axis

    def inside_bounds(self) -> bool:
        return np.all((self.state.cell >= 0) & (self.state.cell < self.res))

    def current_cell(self) -> np.ndarray:
        return self.state.cell

    def current_t(self) -> float:
        return self.state.t
