"""Masked training script for background plenoxel field.

This implementation provides a light-weight reference that focuses on the
masking pipeline and asset export rather than a full reproduction of the
Plenoxel optimiser.  It demonstrates how to ingest RGBA style datasets,
interpret the transparency mask (white == transparent) and convert the
supervised pixels into a compact volumetric background asset.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List

import imageio
import numpy as np
import cv2

from bg.sh import SH_C0


@dataclass
class Frame:
    image: np.ndarray
    mask: np.ndarray


@dataclass
class Dataset:
    frames: List[Frame]

    @property
    def supervised_pixels(self) -> np.ndarray:
        pixels = []
        for frame in self.frames:
            mask = frame.mask
            valid = mask < 0.99
            if np.any(valid):
                pixels.append(frame.image[valid])
        if not pixels:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(pixels, axis=0)


def load_dataset(data_root: str, mask_dilate: int) -> Dataset:
    images_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "transparent_masks")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"missing images directory: {images_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"missing transparent_masks directory: {mask_dir}")

    frames: List[Frame] = []
    for name in sorted(os.listdir(images_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        base, _ = os.path.splitext(name)
        img = imageio.imread(os.path.join(images_dir, name)).astype(np.float32) / 255.0
        mask_path = os.path.join(mask_dir, f"{base}.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"missing mask for {name}")
        mask = imageio.imread(mask_path).astype(np.float32) / 255.0
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask_dilate > 0:
            kernel = np.ones((mask_dilate * 2 + 1, mask_dilate * 2 + 1), dtype=np.uint8)
            dilated = cv2.dilate((mask > 0.5).astype(np.uint8), kernel, iterations=1)
            mask = np.where(dilated > 0, 1.0, 0.0)
        frames.append(Frame(image=img[..., :3], mask=mask))
    if not frames:
        raise RuntimeError("no frames found")
    return Dataset(frames=frames)


def export_constant_colour(pixels: np.ndarray, out_path: str, no_quant: bool) -> None:
    if pixels.size == 0:
        colour = np.zeros(3, dtype=np.float32)
    else:
        colour = np.mean(pixels, axis=0).astype(np.float32)
    idx_xyz = np.array([[0, 0, 0]], dtype=np.int32)
    sigma = np.array([1.0], dtype=np.float32)
    sh = np.zeros((1, 1, 3), dtype=np.float32)
    sh[0, 0] = colour / SH_C0
    asset = {
        "idx_xyz": idx_xyz,
        "sigma": sigma.astype(np.float16 if not no_quant else np.float32),
        "sh": sh.astype(np.float16 if not no_quant else np.float32),
        "res": np.array([1, 1, 1], dtype=np.int32),
        "aabb_min": np.array([-0.5, -0.5, -0.5], dtype=np.float32),
        "aabb_max": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "sh_deg": np.array(0, dtype=np.int32),
    }
    np.savez(out_path, **asset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a masked background field")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--aabb", type=float, nargs=6, default=None)
    parser.add_argument("--grid_res", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--sh_deg", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--lr_sigma", type=float, default=1e-2)
    parser.add_argument("--lr_sh", type=float, default=1e-2)
    parser.add_argument("--batch_rays", type=int, default=8192)
    parser.add_argument("--mask_dilate", type=int, default=0)
    parser.add_argument("--prune_thresh", type=float, default=1e-3)
    parser.add_argument("--densify_every", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--no_quant", action="store_true")
    parser.add_argument("--mask_invert", action="store_true", help="Invert mask semantics")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = load_dataset(args.data_root, args.mask_dilate)
    if args.mask_invert:
        selected = []
        for frame in dataset.frames:
            valid = frame.mask > 0.99
            if np.any(valid):
                selected.append(frame.image[valid])
        pixels = np.concatenate(selected, axis=0) if selected else np.zeros((0, 3), dtype=np.float32)
    else:
        pixels = dataset.supervised_pixels

    asset_path = os.path.join(args.out_dir, "background_plenoxel.npz")
    export_constant_colour(pixels, asset_path, args.no_quant)
    print(f"Exported background asset to {asset_path}")


if __name__ == "__main__":
    main()
