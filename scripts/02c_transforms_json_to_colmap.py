"""
Convert UE5/UE4 transforms.json → COLMAP sparse model (cameras.txt + images.txt + points3D.txt).

New capture format (transforms.json) differs from old poses.json: it stores raw UE5 Euler
angles and positions rather than pre-computed c2w matrices.

Coordinate conversion:
  UE5 world: left-handed, X-forward, Y-right, Z-up, centimetres
  COLMAP world: right-handed, Z-up, metres

  Position:   colmap = (ue5_x/100,  -ue5_y/100,  ue5_z/100)
  Rotation:   R_wc_colmap = diag(1,-1,1) @ R_ue5_rh @ T_cam
    where R_ue5_rh = Rot.from_euler('ZYX', [-yaw,-pitch,-roll])  (LH→RH equiv)
    and   T_cam    = [[0,0,1],[1,0,0],[0,-1,0]]                  (UE5cam→OpenCV cam)

Camera intrinsics derived from transforms.json fov + image resolution:
  fx = fy = (width/2) / tan(fov_h/2)
  cx = width/2,  cy = height/2

Usage:
    python3 scripts/02c_transforms_json_to_colmap.py \\
        --transforms /data/block9_center_init/poses/block9_center_h250_fov45_ov70_nadir/transforms.json \\
        --images     /data/block9_center_init/block9_center_h250_fov45_ov70_nadir/rgb \\
        --output     /data/block9_center_colmap/sparse/0

Output (COLMAP text format):
    <output>/cameras.txt
    <output>/images.txt
    <output>/points3D.txt  (empty — known-pose MVS)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

# OCV cam axes in the RH-UE camera frame (X-fwd, Y-left, Z-up).
# Identical to ue_to_colmap.py's C_INV = [[0,0,1],[-1,0,0],[0,-1,0]].
# R_rh (scipy 'ZYX' with negated angles = F@R_ue@F) already encodes the
# world Y-flip, so no separate T_world is applied to the rotation.
_C_INV = np.array([[0, 0, 1],
                   [-1, 0, 0],
                   [0, -1, 0]], dtype=float)


def ue5_to_c2w(roll_deg: float, pitch_deg: float, yaw_deg: float,
               location_cm: list[float]) -> np.ndarray:
    """
    Convert UE5 camera pose to a 4×4 c2w matrix in COLMAP / OpenCV convention.

    UE5 world: left-handed, X-forward, Y-right, Z-up, centimetres.
    COLMAP world: right-handed (Y-flipped), Z-up, metres.
    Camera axes: X=right, Y=down, Z=forward.

    Steps:
      1. Scipy ZYX with negated angles ≡ F@R_ue@F (double Y-flip → RH equivalent).
      2. R_wc = R_rh @ C_INV  (camera convention change: UE5 cam → OpenCV cam).
      3. Position Y-flip for handedness: (x/100, -y/100, z/100).
    """
    R_rh = Rot.from_euler(
        'ZYX', np.radians([-yaw_deg, -pitch_deg, -roll_deg])
    ).as_matrix()
    R_wc = R_rh @ _C_INV

    pos = np.asarray(location_cm, dtype=float) / 100.0
    t = np.array([pos[0], -pos[1], pos[2]])

    c2w = np.eye(4)
    c2w[:3, :3] = R_wc
    c2w[:3,  3] = t
    return c2w


# ---------------------------------------------------------------------------
# Quaternion from rotation matrix (COLMAP convention: QW QX QY QZ)
# ---------------------------------------------------------------------------

def rotmat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    q = Rot.from_matrix(R).as_quat()   # scipy: x, y, z, w
    x, y, z, w = q
    return float(w), float(x), float(y), float(z)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transforms", required=True, type=Path,
                    help="Path to transforms.json from UE5 capture system")
    ap.add_argument("--images", required=True, type=Path,
                    help="Directory containing the captured PNG images")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory for COLMAP sparse model")
    ap.add_argument("--width",  type=int, default=None, help="Override image width")
    ap.add_argument("--height", type=int, default=None, help="Override image height")
    args = ap.parse_args()

    with open(args.transforms) as f:
        data = json.load(f)

    # New capture format stores "hfov"/"vfov"; older one stored "fov".
    fov_h = float(data.get("fov", data.get("hfov")))   # horizontal FOV in degrees
    frames = data["frames"]

    # ---- Detect image resolution -------------------------------------------
    width = args.width
    height = args.height
    if width is None or height is None:
        # Pick from first available image
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            imgs = sorted(args.images.glob(ext))
            if imgs:
                from PIL import Image as _PIL
                with _PIL.open(imgs[0]) as im:
                    if width is None:
                        width = im.width
                    if height is None:
                        height = im.height
                break
        if width is None:
            raise RuntimeError("Could not determine image size; pass --width/--height")

    # ---- Intrinsics ---------------------------------------------------------
    fx = fy = (width / 2.0) / math.tan(math.radians(fov_h / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    print(f"[transforms2colmap] Intrinsics: {width}×{height}  "
          f"fov_h={fov_h}°  fx=fy={fx:.3f}  cx={cx}  cy={cy}")

    args.output.mkdir(parents=True, exist_ok=True)

    # ---- cameras.txt --------------------------------------------------------
    cameras_path = args.output / "cameras.txt"
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    print(f"[transforms2colmap] cameras.txt → {cameras_path}")

    # ---- images.txt ---------------------------------------------------------
    # Sort frames by image filename so IDs match COLMAP feature_extractor order.
    def frame_imgname(fr) -> str:
        return f"{int(fr['frame']):04d}.png"

    frames_sorted = sorted(frames, key=frame_imgname)

    images_path = args.output / "images.txt"
    n_written = 0
    missing: list[str] = []

    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}\n")

        for img_id, fr in enumerate(frames_sorted, start=1):
            img_name = frame_imgname(fr)
            img_file = args.images / img_name
            if not img_file.exists():
                missing.append(img_name)
                continue

            roll, pitch, yaw = fr["rotation"]       # [roll, pitch, yaw] degrees
            loc = fr["location"]                     # [x, y, z] cm UE5

            c2w = ue5_to_c2w(roll, pitch, yaw, loc)

            # COLMAP needs world-to-camera: R_cw = R_wc^T, t_cw = -R_cw @ t_wc
            R_wc = c2w[:3, :3]
            t_wc = c2w[:3, 3]
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc

            qw, qx, qy, qz = rotmat_to_quat(R_cw)
            tx, ty, tz = t_cw

            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} 1 {img_name}\n")
            f.write("\n")   # empty 2D-points line
            n_written += 1

    print(f"[transforms2colmap] images.txt  → {images_path}  ({n_written} images)")
    if missing:
        print(f"[transforms2colmap] WARNING: {len(missing)} images not found: {missing[:5]}")

    # ---- points3D.txt (empty) -----------------------------------------------
    pts_path = args.output / "points3D.txt"
    with open(pts_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")
    print(f"[transforms2colmap] points3D.txt → {pts_path}  (empty — known-pose MVS)")
    print(f"\n[transforms2colmap] Done.  {n_written}/{len(frames)} cameras written.")

    # ---- Summary ------------------------------------------------------------
    print(f"\nCamera grid summary:")
    locs = [fr["location"] for fr in frames_sorted if (args.images / frame_imgname(fr)).exists()]
    xs = [l[0]/100 for l in locs]
    ys = [-l[1]/100 for l in locs]   # COLMAP Y
    zs = [l[2]/100 for l in locs]
    print(f"  COLMAP world X: [{min(xs):.1f}, {max(xs):.1f}] m  span={max(xs)-min(xs):.1f} m")
    print(f"  COLMAP world Y: [{min(ys):.1f}, {max(ys):.1f}] m  span={max(ys)-min(ys):.1f} m")
    print(f"  COLMAP world Z: {zs[0]:.1f} m (altitude)")


if __name__ == "__main__":
    main()
