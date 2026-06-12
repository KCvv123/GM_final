#!/usr/bin/env python3
"""Generate holdout eval plans for 3DGS PSNR/SSIM/LPIPS evaluation.

Protocol:
  Heights: 120m, 60m, 30m
  Pitches: -90 (nadir), -60, -30  (no -30 at 120m)
  8 positions per (height, pitch) bin
  Oblique yaw cycles through 0/90/180/270
  Total: 64 per scene before leakage filter

Leakage filter:
  Merges ALL methods' second-pass viewpoints for the scene.
  Rejects eval poses within MIN_DIST_XY of any train pose at similar height.

Usage:
  python3 scripts/08_generate_eval_plan.py --scene town01
  python3 scripts/08_generate_eval_plan.py --scene town03
  python3 scripts/08_generate_eval_plan.py --scene all
"""

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_HEIGHTS = [120.0, 60.0, 30.0]
EVAL_PITCHES = [-90.0, -60.0, -30.0]
POSITIONS_PER_BIN = 8
OBLIQUE_YAWS = [0.0, 90.0, 180.0, 270.0]
MIN_DIST_XY = 15.0  # metres — reject eval pose if any train pose is closer
MIN_DIST_Z = 10.0   # metres — only compare poses at similar height
IMAGE_W, IMAGE_H, FOV = 1280, 720, 90.0


def load_viewpoints(filepath):
    """Load viewpoints from FuXian format: x,y,z,dx,dy,dz per line."""
    pts = []
    if not os.path.exists(filepath):
        return pts
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            pts.append((x, -y, z))  # negate Y: COLMAP → CARLA
    return pts


def load_samples_extent(scene):
    """Get XY bounding box from surface sample points (scene geometry)."""
    samples_path = os.path.join(REPO, "data", f"{scene}_samples.ply")
    if not os.path.exists(samples_path):
        return None
    xs, ys = [], []
    with open(samples_path) as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                xs.append(float(parts[0]))
                ys.append(-float(parts[1]))  # negate Y: COLMAP right-handed → CARLA left-handed
    if not xs:
        return None
    margin = 30.0
    return (min(xs) + margin, max(xs) - margin,
            min(ys) + margin, max(ys) - margin)


def generate_grid_positions(x_min, x_max, y_min, y_max, n):
    """Generate n positions in a roughly uniform grid within the bbox."""
    # Find grid dimensions that give ~n points
    aspect = (x_max - x_min) / max(y_max - y_min, 1e-6)
    ny = max(1, int(math.sqrt(n / aspect)))
    nx = max(1, int(n / ny))
    # Adjust to get at least n points
    while nx * ny < n:
        nx += 1

    positions = []
    dx = (x_max - x_min) / max(nx, 1)
    dy = (y_max - y_min) / max(ny, 1)
    for ix in range(nx):
        for iy in range(ny):
            x = x_min + dx * (ix + 0.5)
            y = y_min + dy * (iy + 0.5)
            positions.append((x, y))
    return positions[:n]


def is_too_close(ex, ey, ez, train_pts, min_xy, min_z):
    """Check if eval pose (ex,ey,ez) is too close to any train pose."""
    for tx, ty, tz in train_pts:
        if abs(ez - tz) > min_z:
            continue
        dist_xy = math.sqrt((ex - tx) ** 2 + (ey - ty) ** 2)
        if dist_xy < min_xy:
            return True
    return False


def direction_from_pitch_yaw(pitch_deg, yaw_deg):
    """Convert pitch/yaw (degrees) to unit direction vector."""
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    dx = math.cos(p) * math.cos(y)
    dy = math.cos(p) * math.sin(y)
    dz = math.sin(p)
    return (dx, dy, dz)


def generate_eval_plan(scene, output_dir, min_dist_xy=MIN_DIST_XY, min_dist_z=MIN_DIST_Z):
    """Generate eval plan JSON for one scene."""
    out_dir = os.path.join(REPO, "output", scene)

    # Load all second-pass viewpoints across all methods (including hreq variants)
    import glob
    all_train = []
    patterns = [
        f"{scene}_viewpoints.txt",
        f"{scene}_viewpoints_bc.txt",
        f"{scene}_viewpoints_cwc.txt",
        f"{scene}_viewpoints_hreq*_cwc.txt",
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(out_dir, pat))
        for fp in sorted(matches):
            pts = load_viewpoints(fp)
            all_train.extend(pts)
            print(f"  Loaded {len(pts)} train viewpoints from {os.path.basename(fp)}")

    if not all_train:
        print(f"  ERROR: No viewpoints found for {scene}")
        return None

    # Get scene extent from surface sample points (not viewpoints)
    extent = load_samples_extent(scene)
    if extent is None:
        print(f"  ERROR: Cannot load samples PLY for {scene}, falling back to viewpoints")
        xs = [p[0] for p in all_train]
        ys = [p[1] for p in all_train]
        margin = 30.0
        extent = (min(xs) + margin, max(xs) - margin, min(ys) + margin, max(ys) - margin)
    x_min, x_max, y_min, y_max = extent
    print(f"  Scene extent (from samples, with margin): x=[{x_min:.0f},{x_max:.0f}] y=[{y_min:.0f},{y_max:.0f}]")
    print(f"  Total train viewpoints (all methods merged): {len(all_train)}")

    # Generate more candidate positions than needed (oversample for leakage rejection)
    n_candidates = POSITIONS_PER_BIN * 4  # 4x oversample
    candidate_positions = generate_grid_positions(x_min, x_max, y_min, y_max, n_candidates)

    # Build eval bins
    bins = []
    for h in EVAL_HEIGHTS:
        for p in EVAL_PITCHES:
            if h == 120.0 and p == -30.0:
                continue  # skip: 120m + -30° sees horizon, not useful
            bins.append((h, p))

    # For each bin, pick positions that pass leakage filter
    eval_waypoints = []
    stats = {}
    for height, pitch in bins:
        bin_key = f"h{int(height)}_p{int(pitch)}"
        accepted = []
        yaw_idx = 0

        for (cx, cy) in candidate_positions:
            if len(accepted) >= POSITIONS_PER_BIN:
                break
            if is_too_close(cx, cy, height, all_train, min_dist_xy, min_dist_z):
                continue

            # Assign yaw
            if pitch == -90.0:
                yaw = 0.0  # nadir — yaw doesn't matter
            else:
                yaw = OBLIQUE_YAWS[yaw_idx % len(OBLIQUE_YAWS)]
                yaw_idx += 1

            accepted.append({
                "x": round(cx, 3),
                "y": round(cy, 3),
                "z": round(height, 2),
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "bin": bin_key,
            })

        stats[bin_key] = len(accepted)
        eval_waypoints.extend(accepted)
        print(f"  Bin {bin_key}: {len(accepted)}/{POSITIONS_PER_BIN} accepted")

    if not eval_waypoints:
        print("  ERROR: All eval poses filtered out!")
        return None

    # Build gennbv JSON (single group, per_waypoint orientation, 5-tuple)
    waypoints_5tuple = []
    for wp in eval_waypoints:
        waypoints_5tuple.append([wp["x"], wp["y"], wp["z"], wp["pitch"], wp["yaw"]])

    plan = {
        "schema_version": 2,
        "description": f"Holdout eval plan for {scene} — {len(eval_waypoints)} views, "
                        f"heights {EVAL_HEIGHTS}, pitches {EVAL_PITCHES}",
        "globals": {
            "carla_map": "Town01" if scene == "town01" else "Town03",
            "image_width": IMAGE_W,
            "image_height": IMAGE_H,
        },
        "defaults": {
            "fov": FOV,
            "image_width": IMAGE_W,
            "image_height": IMAGE_H,
        },
        "groups": [
            {
                "name": "eval_holdout",
                "route_mode": "waypoints",
                "waypoint_spacing_mode": "none",
                "waypoints": waypoints_5tuple,
                "look_at_base": "per_waypoint",
                "fov": FOV,
                "image_width": IMAGE_W,
                "image_height": IMAGE_H,
            }
        ],
    }

    # Write JSON
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scene}_eval_holdout.json")
    with open(out_path, "w") as f:
        json.dump(plan, f, indent=2)

    # Also write a CSV manifest for easy analysis
    csv_path = os.path.join(output_dir, f"{scene}_eval_manifest.csv")
    with open(csv_path, "w") as f:
        f.write("idx,x,y,z,pitch,yaw,bin\n")
        for i, wp in enumerate(eval_waypoints):
            f.write(f"{i},{wp['x']},{wp['y']},{wp['z']},{wp['pitch']},{wp['yaw']},{wp['bin']}\n")

    print(f"\n  Written: {out_path} ({len(eval_waypoints)} waypoints)")
    print(f"  Written: {csv_path}")
    print(f"  Bin summary: {stats}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate holdout eval plans")
    parser.add_argument("--scene", required=True, choices=["town01", "town03", "all"])
    parser.add_argument("--output-dir", default=os.path.join(REPO, "data", "plans", "eval"))
    parser.add_argument("--min-dist-xy", type=float, default=MIN_DIST_XY)
    parser.add_argument("--min-dist-z", type=float, default=MIN_DIST_Z)
    args = parser.parse_args()

    min_xy = args.min_dist_xy
    min_z = args.min_dist_z

    scenes = ["town01", "town03"] if args.scene == "all" else [args.scene]

    for scene in scenes:
        print(f"\n{'='*60}")
        print(f"Generating eval plan for {scene}")
        print(f"{'='*60}")
        generate_eval_plan(scene, args.output_dir, min_xy, min_z)


if __name__ == "__main__":
    main()
