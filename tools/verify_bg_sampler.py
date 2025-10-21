"""Compare reference renderer against bg.sample() implementation."""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from bg import SampleOpt, Transform, make_plenoxel_field
from bg.aabb import AABB
from bg.io_npz import load_npz_asset
from bg.sh import apply_sh_rgb, eval_sh_basis


def naive_sample(asset, origin, direction, opt: SampleOpt) -> np.ndarray:
    origin = origin.astype(np.float32)
    direction = direction.astype(np.float32)
    direction /= max(np.linalg.norm(direction), 1e-6)
    aabb = AABB(asset.aabb_min, asset.aabb_max)
    hit, t0, t1 = aabb.intersect(origin, direction)
    if not hit:
        return np.zeros(3, dtype=np.float32)
    t_far = min(t1, opt.t_far_max)
    t_start = max(t0, 0.0)
    voxel_size = (asset.aabb_max - asset.aabb_min) / asset.res
    hits = []
    for idx, coord in enumerate(asset.idx_xyz):
        vmin = asset.aabb_min + coord * voxel_size
        vmax = vmin + voxel_size
        cell_aabb = AABB(vmin, vmax)
        c_hit, c0, c1 = cell_aabb.intersect(origin, direction)
        if not c_hit:
            continue
        hits.append((c0, c1, idx))
    hits.sort(key=lambda x: x[0])
    trans = 1.0
    colour = np.zeros(3, dtype=np.float32)
    basis = eval_sh_basis(asset.sh_deg, direction)
    for c0, c1, idx in hits:
        if c0 > t_far:
            break
        dt = max(0.0, min(c1, t_far) - max(c0, t_start))
        if dt <= 0:
            continue
        sigma = float(asset.sigma[idx])
        if sigma <= 0:
            continue
        rgb = apply_sh_rgb(asset.sh[idx], basis)
        alpha = 1.0 - math.exp(-sigma * dt)
        colour += trans * alpha * rgb
        trans *= 1.0 - alpha
        if trans < opt.tau_min:
            break
    return colour


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify background sampler")
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--rays", type=int, default=10000)
    args = parser.parse_args()

    asset = load_npz_asset(args.asset)
    field = make_plenoxel_field(args.asset, Transform(M=np.eye(3, dtype=np.float32), t=np.zeros(3, dtype=np.float32)))
    opt = SampleOpt()
    rng = np.random.default_rng(42)

    diffs = []
    start = time.time()
    for _ in range(args.rays):
        origin = rng.uniform(asset.aabb_min - 0.5, asset.aabb_max + 0.5)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        ref = naive_sample(asset, origin, direction, opt)
        test = field.sample_rgb(origin, direction, opt)
        diffs.append(np.linalg.norm(ref - test))
    elapsed = time.time() - start
    print(f"Avg error: {np.mean(diffs):.6f}  max error: {np.max(diffs):.6f}")
    print(f"Processed {args.rays} rays in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
