"""
Select ORB key images for the first-pass coarse reconstruction.

Paper Section 3.2.1 selects key images before SfM/MVS: ORB features are
matched between the current image and the adjacent key image, and a new key
image is added when overlap is below a threshold. Images with too few features
are discarded.

This script preserves each selected image's relative path under --images, and
optionally filters CARLA poses.json to the same key-image subset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2


def _resolve_frame_name(frame: dict) -> str:
    raw_path = frame.get("image_path", "")
    parts = Path(raw_path).parts
    try:
        idx = parts.index("images")
        return str(Path(*parts[idx + 1:]))
    except (ValueError, TypeError):
        return str(Path(raw_path))


def _iter_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def _load_orb(path: Path, orb, max_side: int):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return orb.detectAndCompute(img, None)


def _match_count(desc_a, desc_b) -> int:
    if desc_a is None or desc_b is None:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_a, desc_b)
    return len(matches)


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def filter_poses(poses_path: Path, selected_rel: set[str], output_path: Path) -> int:
    with open(poses_path) as f:
        poses = json.load(f)

    frames = []
    for frame in poses.get("frames", []):
        if _resolve_frame_name(frame) in selected_rel:
            copied = dict(frame)
            copied["frame_idx"] = len(frames)
            frames.append(copied)

    poses["frames"] = frames
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(poses, f, indent=2)
    return len(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, type=Path,
                    help="Input first-pass image root")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output key-image root")
    ap.add_argument("--poses", type=Path, default=None,
                    help="Optional poses.json to filter")
    ap.add_argument("--match-threshold", type=int, default=450,
                    help="Select current image as a new key image when matches to previous key image are below this count")
    ap.add_argument("--min-features", type=int, default=150,
                    help="Discard images with fewer ORB keypoints than this")
    ap.add_argument("--max-features", type=int, default=4000)
    ap.add_argument("--max-side", type=int, default=1600,
                    help="Downscale images larger than this side length for ORB matching only")
    ap.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    args = ap.parse_args()

    image_paths = _iter_images(args.images)
    if not image_paths:
        raise SystemExit(f"[keyimg] No images found under {args.images}")

    orb = cv2.ORB_create(nfeatures=args.max_features)
    selected: list[Path] = []
    selected_desc = None
    selected_kp_count = 0
    skipped_low_features = 0

    print(f"[keyimg] Scanning {len(image_paths)} images under {args.images}", flush=True)
    for path in image_paths:
        keypoints, desc = _load_orb(path, orb, args.max_side)
        kp_count = len(keypoints)
        if kp_count < args.min_features:
            skipped_low_features += 1
            continue

        if selected_desc is None:
            selected.append(path)
            selected_desc = desc
            selected_kp_count = kp_count
            continue

        matches = _match_count(selected_desc, desc)
        if matches < args.match_threshold:
            selected.append(path)
            selected_desc = desc
            selected_kp_count = kp_count

    if not selected:
        raise SystemExit("[keyimg] No key images selected. Lower --min-features or inspect input images.")

    selected_rel = {str(p.relative_to(args.images)) for p in selected}
    for src in selected:
        rel = src.relative_to(args.images)
        _copy_or_link(src, args.output / rel, args.link_mode)

    print(f"[keyimg] Selected {len(selected)}/{len(image_paths)} key images", flush=True)
    print(f"[keyimg] Discarded {skipped_low_features} low-feature images", flush=True)
    print(f"[keyimg] Last selected keypoint count: {selected_kp_count}", flush=True)
    print(f"[keyimg] Key images -> {args.output}", flush=True)

    if args.poses:
        out_poses = args.output / "poses.json"
        n_frames = filter_poses(args.poses, selected_rel, out_poses)
        print(f"[keyimg] Filtered poses -> {out_poses} ({n_frames} frames)", flush=True)
        if n_frames != len(selected):
            print(f"[keyimg] WARNING: selected images ({len(selected)}) != filtered pose frames ({n_frames})", flush=True)


if __name__ == "__main__":
    main()
