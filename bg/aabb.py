"""Axis aligned bounding box utilities."""
from __future__ import annotations

import numpy as np


class AABB:
    def __init__(self, aabb_min: np.ndarray, aabb_max: np.ndarray):
        self.min = np.asarray(aabb_min, dtype=np.float32)
        self.max = np.asarray(aabb_max, dtype=np.float32)

    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        """Compute ray intersection with the box."""

        origin = origin.astype(np.float32)
        direction = direction.astype(np.float32)
        inv_dir = np.where(direction != 0.0, 1.0 / direction, np.inf)
        t0s = (self.min - origin) * inv_dir
        t1s = (self.max - origin) * inv_dir
        t_min = np.minimum(t0s, t1s)
        t_max = np.maximum(t0s, t1s)
        t0 = float(np.max(t_min))
        t1 = float(np.min(t_max))
        hit = t1 >= max(t0, 0.0)
        return hit, t0, t1

    def to_array(self) -> np.ndarray:
        return np.stack([self.min, self.max], axis=0)
