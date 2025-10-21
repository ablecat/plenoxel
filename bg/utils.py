"""Utility helpers for Plenoxel background fields."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TransformHelper:
    """Helper object providing world <-> field conversions."""

    M: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        self.M = np.asarray(self.M, dtype=np.float32).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float32).reshape(3)
        # Pre-compute inverse for world conversion.
        self._M_inv = np.linalg.inv(self.M)

    def world_to_field_point(self, x: np.ndarray) -> np.ndarray:
        return (self.M @ x.astype(np.float32)) + self.t

    def world_to_field_dir(self, d: np.ndarray) -> np.ndarray:
        return self.M @ d.astype(np.float32)

    def field_to_world_point(self, x: np.ndarray) -> np.ndarray:
        return self._M_inv @ (x.astype(np.float32) - self.t)

    def field_to_world_dir(self, d: np.ndarray) -> np.ndarray:
        return self._M_inv @ d.astype(np.float32)


def compute_voxel_size(aabb_min: np.ndarray, aabb_max: np.ndarray, res: np.ndarray) -> np.ndarray:
    """Return voxel size for each axis."""

    return (np.asarray(aabb_max) - np.asarray(aabb_min)) / np.asarray(res, dtype=np.float32)


def aabb_corners(aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    """Return the 8 corners of an AABB for debugging purposes."""

    amin = np.asarray(aabb_min, dtype=np.float32)
    amax = np.asarray(aabb_max, dtype=np.float32)
    corners = []
    for dx in (0.0, 1.0):
        for dy in (0.0, 1.0):
            for dz in (0.0, 1.0):
                corners.append(amin + (amax - amin) * np.array([dx, dy, dz], dtype=np.float32))
    return np.stack(corners, axis=0)
