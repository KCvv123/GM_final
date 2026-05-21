"""
Convert CARLA poses.json → COLMAP sparse model (cameras.txt + images.txt + points3D.txt).

CARLA capturer saves exact camera-to-world (c2w) matrices in OpenCV convention
(X right, Y down, Z forward → looking into the scene).  COLMAP uses the same
convention for its camera model, so the conversion is straightforward.

This bypasses the SfM mapper entirely: all images get their known poses, and
COLMAP can proceed directly to image_undistorter → patch_match_stereo → stereo_fusion.

Usage:
    python3 scripts/02b_carla_poses_to_colmap.py \\
        --poses  /data/first_pass/images/poses.json \\
        --images /data/first_pass/images/first_pass_nadir_120m \\
        --output /data/colmap_first_pass/sparse/0

Output (COLMAP text format):
    <output>/cameras.txt
    <output>/images.txt
    <output>/points3D.txt   (empty – no 3D points needed for known-pose MVS)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Quaternion from rotation matrix (COLMAP convention: QW QX QY QZ)
# ---------------------------------------------------------------------------

def rotmat_to_quat(R: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z)."""
    m = R
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2][1] - m[1][2]) * s
        y = (m[0][2] - m[2][0]) * s
        z = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return w, x, y, z


def matmul3x3_vec(R: list[list[float]], v: list[float]) -> list[float]:
    return [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    ]


def transpose3x3(R: list[list[float]]) -> list[list[float]]:
    return [[R[j][i] for j in range(3)] for i in range(3)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", required=True, type=Path,
                    help="Path to poses.json from CARLA capturer")
    ap.add_argument("--images", required=True, type=Path,
                    help="Directory containing the captured PNG images")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory for COLMAP sparse model (cameras.txt etc.)")
    args = ap.parse_args()

    with open(args.poses) as f:
        poses = json.load(f)

    intrinsics = poses["intrinsics"]
    frames = poses["frames"]

    model_str = intrinsics.get("model", "PINHOLE")
    width  = int(intrinsics["width"])
    height = int(intrinsics["height"])
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    args.output.mkdir(parents=True, exist_ok=True)

    # ---- cameras.txt (single shared camera) ---------------------------------
    cameras_path = args.output / "cameras.txt"
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        # PINHOLE: fx fy cx cy
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    print(f"[poses2colmap] cameras.txt → {cameras_path}")

    # ---- images.txt ---------------------------------------------------------
    # COLMAP format per image:
    #   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    #   (empty 2D points line)
    #
    # poses.json gives c2w (camera-to-world, OpenCV convention).
    # COLMAP needs w2c: R_cw = R_wc^T,  t_cw = -R_cw @ t_wc
    images_path = args.output / "images.txt"
    n_written = 0
    missing = []

    # Sort frames by resolved image name so IDs match COLMAP feature_extractor's
    # alphabetical discovery order.
    def _resolve_img_name(frame):
        raw_parts = Path(frame["image_path"]).parts
        try:
            idx = raw_parts.index("images")
            return str(Path(*raw_parts[idx + 1:]))
        except (ValueError, TypeError):
            return str(Path(frame["image_path"]))

    frames_sorted = sorted(frames, key=_resolve_img_name)

    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}\n")

        for img_id, frame in enumerate(frames_sorted, start=1):
            # Resolve image filename relative to --images dir.
            # poses.json paths look like /data/output/images/group/frame_NNNN.png
            # We need the path relative to --images dir (e.g. group/frame_NNNN.png)
            # to match what COLMAP feature_extractor stores in its database.
            raw_path = frame["image_path"]
            raw_parts = Path(raw_path).parts
            # Find "images" in the path and take everything after it
            try:
                img_idx = raw_parts.index("images")
                img_name = str(Path(*raw_parts[img_idx + 1:]))
            except (ValueError, TypeError):
                img_name = str(Path(raw_path))
            img_file = args.images / img_name

            if not img_file.exists():
                missing.append(img_name)
                continue

            c2w = frame["c2w"]  # 4×4 list-of-lists
            R_wc = [c2w[i][:3] for i in range(3)]   # 3×3 rotation (world←camera)
            t_wc = [c2w[i][3]  for i in range(3)]   # camera position in world

            R_cw = transpose3x3(R_wc)               # world-to-camera rotation
            t_cw = matmul3x3_vec(R_cw, [-t_wc[0], -t_wc[1], -t_wc[2]])

            qw, qx, qy, qz = rotmat_to_quat(R_cw)
            tx, ty, tz = t_cw

            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} 1 {img_name}\n")
            f.write("\n")   # empty 2D-points line
            n_written += 1

    print(f"[poses2colmap] images.txt  → {images_path}  ({n_written} images)")
    if missing:
        print(f"[poses2colmap] WARNING: {len(missing)} image files not found: {missing[:5]}")

    # ---- points3D.txt (empty) -----------------------------------------------
    pts_path = args.output / "points3D.txt"
    with open(pts_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")
    print(f"[poses2colmap] points3D.txt → {pts_path}  (empty – known-pose MVS)")

    print(f"\n[poses2colmap] Done.  {n_written}/{len(frames)} cameras written.")
    print(f"  Next: run image_undistorter with --input_path {args.output}")


if __name__ == "__main__":
    main()
