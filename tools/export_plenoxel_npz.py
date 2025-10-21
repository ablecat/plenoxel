"""Utility script to assemble Plenoxel assets from numpy arrays."""
from __future__ import annotations

import argparse
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Plenoxel NPZ asset")
    parser.add_argument("--idx", type=str, required=True, help="Path to idx_xyz .npy file")
    parser.add_argument("--sigma", type=str, required=True, help="Path to sigma .npy file")
    parser.add_argument("--sh", type=str, required=True, help="Path to SH coefficients .npy file")
    parser.add_argument("--res", type=int, nargs=3, required=True)
    parser.add_argument("--aabb", type=float, nargs=6, required=True)
    parser.add_argument("--sh_deg", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--sigma_scale", type=float, default=None)
    parser.add_argument("--sh_scale", type=float, default=None)
    parser.add_argument("--occ", type=str, default=None, help="Optional occupancy pyramid npy")
    args = parser.parse_args()

    asset = {
        "idx_xyz": np.load(args.idx).astype(np.int32),
        "sigma": np.load(args.sigma),
        "sh": np.load(args.sh),
        "res": np.asarray(args.res, dtype=np.int32),
        "aabb_min": np.asarray(args.aabb[:3], dtype=np.float32),
        "aabb_max": np.asarray(args.aabb[3:], dtype=np.float32),
        "sh_deg": np.asarray(args.sh_deg, dtype=np.int32),
    }
    if args.sigma_scale is not None:
        asset["sigma_scale"] = np.asarray(args.sigma_scale, dtype=np.float32)
    if args.sh_scale is not None:
        asset["sh_scale"] = np.asarray(args.sh_scale, dtype=np.float32)
    if args.occ is not None:
        asset["occ_pyr"] = np.load(args.occ).astype(np.uint8)
    np.savez(args.out, **asset)
    print(f"Saved asset to {args.out}")


if __name__ == "__main__":
    main()
