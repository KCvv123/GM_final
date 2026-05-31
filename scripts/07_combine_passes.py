#!/usr/bin/env python3
"""
Combine a first-pass capture plan with a FuXian-generated second-pass plan
into a single JSON that gennbv_blackwall can execute end-to-end.

Both inputs must be schema_version=2 CARLA capture plans. The combined
plan keeps the first-pass globals/defaults (the first-pass file is
usually authoritative on map_name, server settings, grid defaults) and
concatenates groups: first-pass groups first (so the coarse model gets
captured first), then the second-pass FuXian groups.

Globals: required keys (map_name, image_width, image_height, fov) must
match between the two inputs -- a mismatch would make the second-pass
images incompatible with the first-pass coarse reconstruction, which is
the whole point of the two-pass pipeline.

Usage:
    python3 scripts/07_combine_passes.py \\
        --first-pass  /path/to/ov70_nadir_y.json \\
        --second-pass /path/to/second_pass_plan_cwc.json \\
        --output      /path/to/combined_plan_cwc.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


REQUIRED_GLOBAL_KEYS_MUST_MATCH = ("map_name", "image_width", "image_height", "fov")


def load_plan(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"[combine] ERROR: plan not found: {path}")
    with path.open() as f:
        return json.load(f)


def check_schema(plan: dict, label: str) -> None:
    if plan.get("schema_version") != 2:
        sys.exit(f"[combine] ERROR: {label} schema_version != 2: got {plan.get('schema_version')!r}")
    if "groups" not in plan:
        sys.exit(f"[combine] ERROR: {label} has no 'groups' field")


def check_compatible_globals(fp: dict, sp: dict) -> None:
    fg = fp.get("globals", {})
    sg = sp.get("globals", {})
    for k in REQUIRED_GLOBAL_KEYS_MUST_MATCH:
        if k in fg and k in sg and fg[k] != sg[k]:
            sys.exit(f"[combine] ERROR: globals.{k} differs — first-pass={fg[k]!r} second-pass={sg[k]!r}. "
                     f"Refusing to combine; the two passes must share the same camera intrinsics + map.")


def disambiguate_group_names(fp_groups: list, sp_groups: list) -> tuple[list, list]:
    """If the two plans share a group name, prefix the second-pass ones."""
    fp_names = {g.get("name") for g in fp_groups if g.get("name")}
    sp_groups_out = []
    for g in sp_groups:
        g = copy.deepcopy(g)
        name = g.get("name", "")
        if name in fp_names:
            g["name"] = f"sp__{name}"
        sp_groups_out.append(g)
    return [copy.deepcopy(g) for g in fp_groups], sp_groups_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-pass",  type=Path, required=True,
                    help="Path to the first-pass plan JSON (e.g. ov70_nadir_y.json).")
    ap.add_argument("--second-pass", type=Path, required=True,
                    help="Path to the second-pass plan JSON from 05_fuxian_to_carla_plan.py.")
    ap.add_argument("--output",      type=Path, required=True,
                    help="Where to write the combined plan JSON.")
    args = ap.parse_args()

    fp = load_plan(args.first_pass)
    sp = load_plan(args.second_pass)
    check_schema(fp, "first-pass")
    check_schema(sp, "second-pass")
    check_compatible_globals(fp, sp)

    combined = copy.deepcopy(fp)
    fp_groups, sp_groups = disambiguate_group_names(fp.get("groups", []),
                                                    sp.get("groups", []))
    combined["groups"] = fp_groups + sp_groups

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(combined, f, indent=2)

    n_fp = len(fp_groups)
    n_sp = len(sp_groups)
    n_fp_wp = sum(len(g.get("waypoints", [])) for g in fp_groups)
    n_sp_wp = sum(len(g.get("waypoints", [])) for g in sp_groups)
    print(f"[combine] Wrote {args.output}")
    print(f"  first-pass:  {n_fp} groups, {n_fp_wp} explicit waypoints"
          + ("  (route_mode=grid, gennbv computes the grid)" if n_fp_wp == 0 else ""))
    print(f"  second-pass: {n_sp} groups, {n_sp_wp} explicit waypoints")
    print(f"  combined:    {n_fp + n_sp} groups")
    print()
    print(f"[combine] To use in gennbv_blackwall:")
    print(f"  NBV_CARLA_COLMAP_CAPTURE_PLAN_PATH={args.output}")
    map_name = combined.get("globals", {}).get("map_name", "Town01")
    print(f"  NBV_CARLA_COLMAP_MAP={map_name}")


if __name__ == "__main__":
    main()
