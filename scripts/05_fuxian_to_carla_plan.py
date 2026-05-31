"""
Convert FuXian (117-paper) output viewpoints → gennbv_blackwall CARLA capture plan JSON.

The FuXian binary writes viewpoints to:
  output/<tag>/<name>.txt
  Format per line:  x,y,z,dx,dy,dz
  where (x,y,z) = camera world position, (dx,dy,dz) = look-at direction vector.

This script converts those viewpoints into a CARLA V2 capture plan JSON that
can be passed to NBVCarlaColmap3DGSPipeline via:
  NBV_CARLA_COLMAP_CAPTURE_PLAN_PATH=/data/path_setting/fuxian_town01_plan.json

Grouping strategy:
  By default, viewpoints are sorted by absolute pitch angle.  Groups are formed by quantising
  each viewpoint's pitch to the nearest PITCH_BUCKET_DEG degrees.  Within each
  group the waypoints are ordered by a nearest-neighbour TSP heuristic so the
  drone flies an efficient path.

  For the paper Section 3.4 ACO path, pass --preserve-order so this converter
  does not reorder the already optimized path. Pass --one-group-per-waypoint to
  preserve the per-viewpoint z coordinate using the existing CARLA plan schema.

Usage:
  python3 scripts/fuxian_to_carla_plan.py \\
      --input  output/town01/town01_viewpoints.txt \\
      --output /data/path_setting/fuxian_town01_plan.json \\
      --map    Town01

  # optional overrides
      --pitch-bucket 15   # group by 15-degree pitch bands (default 22.5)
      --fov        90     # camera horizontal FOV (default 90)
      --width      1280   # image width  (default 1280)
      --height     720    # image height (default 720)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_viewpoints(path: Path) -> list[dict]:
    """Parse FuXian viewpoint text file. Each line: x,y,z,dx,dy,dz"""
    vps = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parts = [float(v) for v in line.split(",")]
                if len(parts) != 6:
                    print(f"[WARN] line {lineno}: expected 6 values, got {len(parts)} – skipped",
                          flush=True)
                    continue
                x, y, z, dx, dy, dz = parts
                # FuXian works in COLMAP right-handed coords, but CARLA
                # waypoints are in UE4 left-handed coords.  The capturer's
                # carla_to_opencv_c2w() negates the Y row of the c2w matrix
                # when saving poses.json, so the COLMAP reconstruction has
                # Y flipped relative to the CARLA world.  Undo that here.
                carla_x =  x
                carla_y = -y   # negate Y: right-handed → left-handed
                carla_z =  z

                carla_dx =  dx
                carla_dy = -dy  # negate direction Y as well
                carla_dz =  dz

                # Compute pitch and yaw from direction vector (in CARLA coords)
                dist_horiz = math.sqrt(carla_dx*carla_dx + carla_dy*carla_dy)
                pitch_deg  = math.degrees(math.atan2(carla_dz, dist_horiz))
                # pitch < 0 = looking down (typical aerial)
                yaw_deg    = math.degrees(math.atan2(carla_dy, carla_dx))
                vps.append({
                    "x": carla_x, "y": carla_y, "z": carla_z,
                    "dx": carla_dx, "dy": carla_dy, "dz": carla_dz,
                    "pitch_deg": pitch_deg,
                    "yaw_deg":   yaw_deg,
                })
            except ValueError as e:
                print(f"[WARN] line {lineno}: parse error ({e}) – skipped", flush=True)
    return vps


def _nearest_neighbour_order(points: list[dict]) -> list[dict]:
    """Simple nearest-neighbour tour for a list of 2-D (x,y) waypoints."""
    if len(points) <= 2:
        return list(points)
    remaining = list(points)
    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]
        best_i = min(
            range(len(remaining)),
            key=lambda i: (remaining[i]["x"]-last["x"])**2 + (remaining[i]["y"]-last["y"])**2,
        )
        ordered.append(remaining.pop(best_i))
    return ordered


def _quantise_pitch(pitch_deg: float, bucket: float) -> float:
    """Round pitch to nearest bucket midpoint."""
    return round(pitch_deg / bucket) * bucket


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="FuXian viewpoints text file")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output CARLA capture plan JSON path")
    ap.add_argument("--map", default="Town01",
                    help="CARLA map name (default: Town01)")
    ap.add_argument("--pitch-bucket", type=float, default=22.5,
                    help="Group viewpoints into pitch bands of this width in degrees "
                         "(default: 22.5).  Use 0 to create one group per viewpoint "
                         "(not recommended).")
    ap.add_argument("--fov",    type=float, default=90.0,  help="Camera FOV (default 90)")
    ap.add_argument("--width",  type=int,   default=1280,  help="Image width (default 1280)")
    ap.add_argument("--height", type=int,   default=720,   help="Image height (default 720)")
    ap.add_argument("--quality", default="Low",
                    help="CARLA render quality: Low / Medium / High (default: Low)")
    ap.add_argument("--preserve-order", action="store_true",
                    help="Keep input order instead of pitch sorting / nearest-neighbour reorder")
    ap.add_argument("--one-group-per-waypoint", action="store_true",
                    help="Emit one CARLA group per viewpoint so each viewpoint keeps its "
                         "own z. Each group becomes look_at_base=fixed_angle + fixed_pitch/yaw. "
                         "DEPRECATED — use --single-group-per-waypoint-orientation instead "
                         "(equivalent capture, but gennbv visualises the connected path).")
    ap.add_argument("--single-group-per-waypoint-orientation", action="store_true",
                    dest="single_group_per_waypoint",
                    help="Emit ONE CARLA group with [x,y,z,pitch,yaw] 5-tuples and "
                         "look_at_base=per_waypoint. Requires gennbv built after "
                         "2026-05-31 (per_waypoint orientation support). gennbv "
                         "will visualise the full ACO path because all viewpoints "
                         "live in one group.")
    args = ap.parse_args()

    print(f"[conv] Reading viewpoints from {args.input}", flush=True)
    vps = _parse_viewpoints(args.input)
    if not vps:
        print("[ERROR] No viewpoints parsed. Check input file.", flush=True)
        sys.exit(1)

    print(f"[conv] Parsed {len(vps)} viewpoints", flush=True)

    # ---- Compute flight height distribution --------------------------------
    zvals = [v["z"] for v in vps]
    z_mean = sum(zvals) / len(zvals)
    z_min  = min(zvals)
    z_max  = max(zvals)
    print(f"[conv] Z range: {z_min:.1f} – {z_max:.1f} m  (mean {z_mean:.1f} m)", flush=True)

    # ---- Group by quantised pitch, with per-waypoint orientation -------------
    if args.single_group_per_waypoint:
        # All viewpoints in ONE group; each waypoint carries its own z + pitch + yaw.
        groups = {0.0: list(vps)}
    elif args.one_group_per_waypoint:
        groups = {float(i): [vp] for i, vp in enumerate(vps)}
    elif args.preserve_order:
        groups = {0.0: list(vps)}
    elif args.pitch_bucket > 0:
        groups: dict[float, list[dict]] = {}
        for vp in vps:
            key = _quantise_pitch(vp["pitch_deg"], args.pitch_bucket)
            groups.setdefault(key, []).append(vp)
    else:
        groups = {0.0: list(vps)}

    mode = "single group, per_waypoint orientation" if args.single_group_per_waypoint \
        else "one group per waypoint" if args.one_group_per_waypoint \
        else "preserve input order" if args.preserve_order \
        else f"pitch groups ({args.pitch_bucket}° bands)"
    print(f"[conv] Groups ({mode}):", flush=True)
    for pitch_key in sorted(groups):
        g = groups[pitch_key]
        z_avg = sum(v["z"] for v in g) / len(g)
        label = f"idx={int(pitch_key):04d}" if args.one_group_per_waypoint else f"pitch≈{pitch_key:+.1f}°"
        print(f"  {label}  {len(g):3d} viewpoints  z_avg={z_avg:.1f} m",
              flush=True)

    # ---- Build CARLA plan ---------------------------------------------------
    # Two output schemes depending on flags:
    #
    #   --single-group-per-waypoint-orientation: ONE group, [x,y,z,pitch,yaw]
    #       5-tuples, look_at_base="per_waypoint". Lossless AND gennbv
    #       visualises the connected path because all viewpoints share a group.
    #       Requires gennbv with the 2026-05-31 per_waypoint patch.
    #
    #   Otherwise: 2-tuple [x,y] waypoints + look_at_base="fixed_angle" +
    #       group-level fixed_pitch/fixed_yaw. Lossless only when each group
    #       has 1 waypoint (--one-group-per-waypoint). Other multi-waypoint
    #       groups use median pitch/yaw with a warning.
    plan_groups = []
    group_keys = list(groups.keys()) if (args.preserve_order or args.one_group_per_waypoint
                                          or args.single_group_per_waypoint) \
        else sorted(groups, key=lambda k: abs(k))
    for pitch_key in group_keys:
        g = groups[pitch_key]

        z_sorted = sorted(v["z"] for v in g)
        flight_z = z_sorted[len(z_sorted) // 2]

        ordered = list(g) if (args.preserve_order or args.one_group_per_waypoint
                              or args.single_group_per_waypoint) \
            else _nearest_neighbour_order(g)

        if args.single_group_per_waypoint:
            # 5-tuple [x,y,z,pitch,yaw] — per-waypoint everything, gennbv
            # picks z from wp[2] and orientation from wp[3:5].
            waypoints = [
                [round(v["x"], 3), round(v["y"], 3), round(v["z"], 2),
                 round(v["pitch_deg"], 2), round(v["yaw_deg"], 2)]
                for v in ordered
            ]
            group_name = "fuxian_path"
            group = {
                "name":            group_name,
                "enabled":         True,
                "route_mode":      "waypoints",
                "waypoint_spacing_mode": "none",
                "waypoints":       waypoints,
                "flight_height":   round(flight_z, 2),  # ignored by gennbv when wp has z
                "look_at_base":    "per_waypoint",
                "fov":             args.fov,
                "image_width":     args.width,
                "image_height":    args.height,
            }
        else:
            waypoints = [
                [round(v["x"], 3), round(v["y"], 3)] for v in ordered
            ]
            if args.one_group_per_waypoint:
                # Lossless: 1 waypoint per group, take its exact pitch/yaw.
                first = ordered[0]
                fixed_pitch = round(first["pitch_deg"], 2)
                fixed_yaw   = round(first["yaw_deg"], 2)
            else:
                # Multi-waypoint group with fixed_angle is approximate -- one
                # (pitch, yaw) covers the whole group. We use the median.
                pitches = sorted(v["pitch_deg"] for v in ordered)
                yaws    = sorted(v["yaw_deg"]   for v in ordered)
                fixed_pitch = round(pitches[len(pitches)//2], 2)
                fixed_yaw   = round(yaws[len(yaws)//2], 2)
                if len(ordered) > 1:
                    yaw_span = yaws[-1] - yaws[0]
                    if yaw_span > 1.0:
                        print(f"[warn] Group {pitch_key!r} has {len(ordered)} waypoints "
                              f"spanning {yaw_span:.1f}° in yaw; gennbv's fixed_angle "
                              "uses one yaw for the whole group. Consider "
                              "--single-group-per-waypoint-orientation for lossless emission.")

            group_name = f"fuxian_wp_{int(pitch_key):04d}" if args.one_group_per_waypoint \
                else f"fuxian_path_{int(pitch_key):04d}" if args.preserve_order \
                else f"fuxian_pitch_{pitch_key:+.0f}deg"
            group = {
                "name":            group_name,
                "enabled":         True,
                "route_mode":      "waypoints",
                "waypoint_spacing_mode": "none",
                "waypoints":       waypoints,
                "flight_height":   round(flight_z, 2),
                "look_at_base":    "fixed_angle",
                "fixed_pitch":     fixed_pitch,
                "fixed_yaw":       fixed_yaw,
                "fov":             args.fov,
                "image_width":     args.width,
                "image_height":    args.height,
            }
        plan_groups.append(group)

    plan = {
        "schema_version": 2,
        "globals": {
            "map_name":     args.map,
            "image_width":  args.width,
            "image_height": args.height,
            "fov":          args.fov,
            "quality_level": args.quality,
            "launch_server": True,
            "server_host":   "localhost",
            "rpc_port":      2001,
            "force_load_world": True,
        },
        "defaults": {},
        "groups": plan_groups,
    }

    # ---- Write output -------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(plan, f, indent=2)

    total_wps = sum(len(g["waypoints"]) for g in plan_groups)
    print(f"\n[conv] Written {len(plan_groups)} groups, "
          f"{total_wps} total waypoints → {args.output}", flush=True)
    print("[conv] To use in gennbv_blackwall, set:", flush=True)
    print(f"  NBV_CARLA_COLMAP_CAPTURE_PLAN_PATH={args.output}", flush=True)
    print(f"  NBV_CARLA_COLMAP_MAP={args.map}", flush=True)


if __name__ == "__main__":
    main()
