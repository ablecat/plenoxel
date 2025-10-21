"""Unified background field interface for Plenoxel assets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import io_npz
from .utils import TransformHelper


@dataclass
class SampleOpt:
    """Sampling options controlling short-range volume rendering."""

    t_near_bias: float = 1e-3
    t_far_max: float = 1e4
    max_steps: int = 16
    tau_min: float = 1e-3
    lod: int = -1
    linear_rgb: bool = True


@dataclass
class Transform:
    """World to field affine transform."""

    M: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        self.M = np.asarray(self.M, dtype=np.float32).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float32).reshape(3)


class IBgField(Protocol):
    """Background field protocol."""

    def aabb_world(self) -> np.ndarray:
        """Return world-space axis aligned bounding box (min, max)."""

    def is_ready(self) -> bool:
        """Return whether the field is fully initialised."""

    def sample_rgb(
        self, x_world: np.ndarray, dir_world: np.ndarray, opt: SampleOpt
    ) -> np.ndarray:
        """Sample RGB colour from the field."""


def make_plenoxel_field(npz_path: str, world_to_field: Transform) -> IBgField:
    """Create a Plenoxel backed background field from an exported asset."""

    asset = io_npz.load_npz_asset(npz_path)
    helper = TransformHelper(world_to_field.M, world_to_field.t)
    from .plenoxel_field import PlenoxelField

    return PlenoxelField(asset=asset, transform=helper)
