#!/usr/bin/env python3
"""
Step: Evaluate final reconstructions of multiple methods — PAPER-FAITHFUL,
no ground truth needed (Remote Sens. 2021, 13, 989, §4.1.3 "self-collected").

The paper evaluates self-collected / virtual scenes WITHOUT external GT, using:
  * accuracy     = §3.2.3 quality score g = gc * gs, normalized (Eq 1-5)
  * completeness = ICP-align (skipped here: all methods share the CARLA world
                   frame) -> equal-size voxels -> voxel-occupancy percentage

This is the metric that REPLACES the (wrong-for-this-paper) 3DGS PSNR/SSIM/LPIPS.

Key correctness points vs a naive reuse of 03_colmap_mesh_to_fuxian_input.py:
  - 03.compute_quality_scores() min-max normalizes WITHIN one cloud -> would
    wash out cross-method differences (every method ~0.5). Here we keep the
    RAW g per sample and apply ONE unified min-max across ALL methods, exactly
    as the paper's "unified normalization ... on the same scale".
  - Sampling is on the Poisson iso-surface (meshed-poisson.ply) to match the
    planner. If a mesh is not supplied we fall back to Poisson-disk subsampling
    of the fused cloud (less faithful; a warning is printed).

Usage:
  python3 scripts/09_evaluate_methods.py \
      --method bc=/path/UE/bc/colmap \
      --method cwc=/path/UE/cwc/colmap \
      --method fuxian=/path/UE/fuxian/colmap \
      --out results_gm_UE.csv

  Each <workspace> must contain fused.ply (and ideally meshed-poisson.ply).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# Reuse the EXACT readers/sampler from step 03 so geometry handling is identical.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
_m3 = importlib.import_module("03_colmap_mesh_to_fuxian_input")
read_ply_mesh = _m3.read_ply_mesh
poisson_disk_sample_mesh = _m3.poisson_disk_sample_mesh
read_dense_cloud = _m3.read_dense_cloud


# ---------------------------------------------------------------------------
# Quality scoring — RAW gc/gs/g (Paper Eq 1-5). Math copied verbatim from
# 03.compute_quality_scores(), but returns RAW values (no per-cloud min-max),
# so the caller can apply ONE unified normalization across methods.
# ---------------------------------------------------------------------------
def compute_raw_quality(samples, dense_pos, dense_normals,
                        radius=3.0, theta_t_deg=60.0):
    from scipy.spatial import cKDTree

    tree = cKDTree(dense_pos)
    sample_pos = np.array([(s[0], s[1], s[2]) for s in samples], dtype=np.float32)
    sample_normals = np.array([(s[3], s[4], s[5]) for s in samples], dtype=np.float64)

    dense_normals = np.asarray(dense_normals, dtype=np.float64)
    dn = np.linalg.norm(dense_normals, axis=1, keepdims=True)
    dense_normals = dense_normals / np.maximum(dn, 1e-12)
    sn = np.linalg.norm(sample_normals, axis=1, keepdims=True)
    sample_normals = sample_normals / np.maximum(sn, 1e-12)

    neighbor_lists = tree.query_ball_point(sample_pos, radius)
    n = len(samples)
    gc = np.zeros(n); gs = np.zeros(n)
    n_max = max((len(nl) for nl in neighbor_lists), default=0)
    theta_t = math.radians(theta_t_deg)
    theta_denom = max((1.0 - math.cos(theta_t)) ** 2, 1e-12)

    for i, nbr in enumerate(neighbor_lists):
        if not nbr or n_max <= 0:
            continue
        nbr_pos = dense_pos[nbr]; nbr_nrm = dense_normals[nbr]
        n_i = sample_normals[i]
        d = np.sqrt(np.sum((nbr_pos - sample_pos[i]) ** 2, axis=1))
        w_pj = np.exp(-d / max(radius, 1e-12))
        grad = np.sum(w_pj[:, None] * nbr_nrm, axis=0)
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm
        count = len(nbr)
        w_i = (1.0 - ((count - n_max) / n_max)) ** 2
        gc[i] = w_i * max(0.0, float(np.dot(grad, n_i)))
        dots = np.clip(nbr_nrm @ n_i, -1.0, 1.0)
        gs[i] = float(np.sum(w_pj * np.exp(-((1.0 - dots) ** 2) / theta_denom)))

    return gc, gs, gc * gs


# ---------------------------------------------------------------------------
def parse_crop(spec: str):
    """Parse 'xmin:xmax,ymin:ymax,zmin:zmax' into a (3,2) float array."""
    try:
        bounds = []
        for axis in spec.split(","):
            lo, hi = axis.split(":")
            bounds.append((float(lo), float(hi)))
        if len(bounds) != 3:
            raise ValueError
        return np.asarray(bounds)
    except ValueError:
        sys.exit(f"--crop must be 'xmin:xmax,ymin:ymax,zmin:zmax', got {spec!r}")


def crop_mask(pos: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    m = np.ones(len(pos), dtype=bool)
    for a in range(3):
        m &= (pos[:, a] >= bounds[a, 0]) & (pos[:, a] <= bounds[a, 1])
    return m


def sample_surface(ws: Path, num_samples: int, rng,
                   cloud_pos=None, cloud_nrm=None,
                   sample_voxel: float = 2.0) -> tuple[list, str]:
    """Return surface sample points (x,y,z,nx,ny,nz), voxel-uniform over the
    fused cloud: one candidate point per occupied voxel, then num_samples
    voxels drawn uniformly. Unlike naive random subsampling (density-weighted,
    oversamples dense regions) this weights every occupied region equally —
    the same spatial uniformity the Poisson-mesh sampling would provide,
    without depending on mesher/trimmer settings.
    """
    if cloud_pos is None:
        cloud_pos, cloud_nrm = read_dense_cloud(ws / "fused.ply")
    pos, nrm = cloud_pos, cloud_nrm
    keys = np.floor(pos / sample_voxel).astype(np.int64)
    # One representative point per voxel (first occurrence after a shuffle so
    # the within-voxel pick is random but reproducible via rng).
    order = rng.permutation(len(pos))
    _, first_idx = np.unique(keys[order], axis=0, return_index=True)
    rep = order[first_idx]                      # one point index per voxel
    k = min(num_samples, len(rep))
    chosen = rng.choice(rep, size=k, replace=False)
    samples = [(float(pos[j, 0]), float(pos[j, 1]), float(pos[j, 2]),
                float(nrm[j, 0]), float(nrm[j, 1]), float(nrm[j, 2]))
               for j in chosen]
    return samples, f"voxel-uniform({len(rep)} cells)"


def voxel_set(pos: np.ndarray, origin: np.ndarray, vsize: float) -> set:
    keys = np.floor((pos - origin) / vsize).astype(np.int64)
    return set(map(tuple, keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", required=True,
                    metavar="NAME=WORKSPACE",
                    help="Repeatable. WORKSPACE holds fused.ply [+ meshed-poisson.ply]")
    ap.add_argument("--out", type=Path, default=Path("results_methods.csv"))
    ap.add_argument("--num-samples", type=int, default=4000)
    ap.add_argument("--quality-radius", type=float, default=3.0)
    ap.add_argument("--theta-t-deg", type=float, default=60.0)
    ap.add_argument("--voxel-size", type=float, default=2.0,
                    help="Voxel edge (metres) for completeness occupancy")
    ap.add_argument("--crop", type=str, default=None,
                    metavar="xmin:xmax,ymin:ymax,zmin:zmax",
                    help="Scene bounds. MVS clouds contain far-away spurious "
                         "points; without cropping each one occupies its own "
                         "voxel and inflates completeness for noisier clouds.")
    ap.add_argument("--norm-pct", type=float, default=100.0,
                    help="Percentile for unified normalization bounds "
                         "(100 = exact min/max as Paper Eq 5; 99 = robust, "
                         "ignores the top/bottom 1%% tail)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    crop_bounds = parse_crop(args.crop) if args.crop else None

    methods = []
    for spec in args.method:
        if "=" not in spec:
            sys.exit(f"--method must be NAME=WORKSPACE, got {spec!r}")
        name, ws = spec.split("=", 1)
        methods.append((name, Path(ws)))

    rng = np.random.default_rng(args.seed)
    per = {}          # name -> dict of results
    raw_g_all = []    # pooled raw g for unified normalization
    fused_pos = {}    # name -> Nx3 for voxel completeness

    for name, ws in methods:
        fused = ws / "fused.ply"
        if not fused.is_file():
            print(f"[{name}] MISSING {fused} — skipped"); continue
        print(f"\n=== {name}  ({ws}) ===", flush=True)
        pos, nrm = read_dense_cloud(fused)
        if crop_bounds is not None:
            m = crop_mask(pos, crop_bounds)
            n_out = int((~m).sum())
            print(f"  [crop] dropped {n_out} outlier points "
                  f"({100.0 * n_out / max(len(pos), 1):.2f}%)", flush=True)
            pos, nrm = pos[m], nrm[m]
        fused_pos[name] = pos
        samples, src = sample_surface(ws, args.num_samples, rng,
                                      cloud_pos=pos, cloud_nrm=nrm)
        gc, gs, g = compute_raw_quality(samples, pos, nrm,
                                        args.quality_radius, args.theta_t_deg)
        raw_g_all.append(g)
        per[name] = dict(points=len(pos), samples=len(samples), src=src,
                         raw_g=g, mean_gc=float(gc.mean()), mean_gs=float(gs.mean()),
                         mean_raw_g=float(g.mean()))
        print(f"  points={len(pos)}  samples={len(samples)} ({src})  "
              f"mean raw g={g.mean():.4f}", flush=True)

    if not per:
        sys.exit("No methods produced results.")

    # ---- ACCURACY: unified min-max across ALL methods' raw g ----------------
    # norm_pct < 100 uses percentile bounds instead of the exact min/max:
    # raw g has a long right tail (a few ultra-dense samples), so exact-max
    # normalization compresses every mean toward 0 and makes the numbers
    # unreadable. Percentile bounds keep ranking/ratios, clip to [0,1].
    pooled = np.concatenate(raw_g_all)
    if args.norm_pct < 100.0:
        gmin = float(np.percentile(pooled, 100.0 - args.norm_pct))
        gmax = float(np.percentile(pooled, args.norm_pct))
    else:
        gmin, gmax = float(pooled.min()), float(pooled.max())
    span = (gmax - gmin) if gmax > gmin else 1.0
    for name in per:
        gn = np.clip((per[name]["raw_g"] - gmin) / span, 0.0, 1.0)
        per[name]["accuracy"] = float(gn.mean())

    # ---- COMPLETENESS: shared voxel grid, occupancy vs union ----------------
    origin = np.min([p.min(0) for p in fused_pos.values()], axis=0)
    vsets = {n: voxel_set(p, origin, args.voxel_size) for n, p in fused_pos.items()}
    union = set().union(*vsets.values()) if vsets else set()
    for name in per:
        occ = len(vsets[name])
        per[name]["occupied_voxels"] = occ
        per[name]["completeness"] = occ / len(union) if union else 0.0

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"{'method':10}{'points':>10}{'accuracy':>11}{'completeness':>14}{'voxels':>9}")
    print("-" * 64)
    rows = []
    for name, _ in methods:
        if name not in per:
            continue
        r = per[name]
        print(f"{name:10}{r['points']:>10}{r['accuracy']:>11.4f}"
              f"{r['completeness']:>14.4f}{r['occupied_voxels']:>9}")
        rows.append((name, r))
    print("=" * 64)
    print(f"(accuracy = unified-normalized §3.2.3 quality, higher=better; "
          f"completeness = voxel occupancy / union @ {args.voxel_size} m)")

    with open(args.out, "w") as f:
        f.write("method,points,samples,sample_src,accuracy,completeness,"
                "occupied_voxels,mean_raw_g,mean_gc,mean_gs\n")
        for name, r in rows:
            f.write(f"{name},{r['points']},{r['samples']},{r['src']},"
                    f"{r['accuracy']:.6f},{r['completeness']:.6f},"
                    f"{r['occupied_voxels']},{r['mean_raw_g']:.6f},"
                    f"{r['mean_gc']:.6f},{r['mean_gs']:.6f}\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
