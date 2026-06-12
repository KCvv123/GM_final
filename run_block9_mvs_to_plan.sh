#!/usr/bin/env bash
#
# run_block9_mvs_to_plan.sh
#
# 117-paper pipeline for block9 (Matrix City / Small_City_LVL).
# Picks up after point_triangulator + image_undistorter (already done).
#
# Steps:
#   1. patch_match_stereo   (GPU, ~60-90 min)
#   2. stereo_fusion        (~5-15 min)
#   3. poisson_mesher       (~2-5 min)
#   4. 03_colmap_mesh_to_fuxian_input.py  (~2 min, Poisson-disk sampling)
#   5. Generate code/utils/Params_block9.h
#   6. 04_run_fuxian.sh  (evolutionary path planning, ~10-30 min)
#   7. 05_fuxian_to_carla_plan.py  → block9_second_pass_plan[_<method>].json
#
# Resumable: each step is skipped if its output already exists.
# Steps 1-5 run once; step 6-7 are per-method, re-run safely for each method.
#
# Usage:
#   cd ~/safe_repos/117-.../
#   bash run_block9_mvs_to_plan.sh                                    # baseline (yan_two_stage)
#   FUXIAN_METHOD=binary_coverage      bash run_block9_mvs_to_plan.sh # bc
#   FUXIAN_METHOD=confidence_coverage  bash run_block9_mvs_to_plan.sh # cwc
#   bash run_block9_mvs_to_plan.sh --skip-mvs                         # skip steps 1-3

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
WORKSPACE=/home/vmlab/safe_repos/block9_v2_colmap
REPO="$(cd "$(dirname "$0")" && pwd)"
DENSE_DIR="$WORKSPACE/dense/0"

COLMAP_IMAGE=vmlab-bin_colmap_38:latest
GPU_INDEX=0

MAP_TAG=block9
MAP_NAME=Small_City_LVL     # UE5 level name for the CARLA capture plan

# Method: yan_two_stage (default/baseline), binary_coverage, confidence_coverage
METHOD="${FUXIAN_METHOD:-yan_two_stage}"
case "$METHOD" in
    confidence_coverage) METHOD_SUFFIX="_cwc" ;;
    binary_coverage)     METHOD_SUFFIX="_bc"  ;;
    *)                   METHOD_SUFFIX=""     ;;
esac

PLAN_OUT="$WORKSPACE/block9_second_pass_plan${METHOD_SUFFIX}.json"

# ── Argument parse ────────────────────────────────────────────────────────────
SKIP_MVS=0
for arg in "$@"; do [[ "$arg" == --skip-mvs ]] && SKIP_MVS=1; done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

die() { echo "[ERROR] $*" >&2; exit 1; }

# Run a COLMAP command inside the Docker container.
# The dense/0 directory is set as CWD inside the container so relative paths work.
colmap_docker() {
    docker run --rm --gpus "device=$GPU_INDEX" \
        -v "$WORKSPACE":/workspace \
        "$COLMAP_IMAGE" \
        bash -c "cd /workspace/dense/0 && $1"
}

# ── Step 1: patch_match_stereo ────────────────────────────────────────────────
if [[ "$SKIP_MVS" -eq 1 ]]; then
    log "Step 1: --skip-mvs set, skipping patch_match_stereo."
elif ls "$DENSE_DIR/stereo/depth_maps/"*.geometric.bin 2>/dev/null | head -1 | grep -q .; then
    log "Step 1: geometric depth maps found, skipping patch_match_stereo."
else
    log "Step 1: patch_match_stereo (geometric consistency) — ~60-90 min ..."
    colmap_docker "colmap patch_match_stereo \
        --workspace_path /workspace/dense/0 \
        --workspace_format COLMAP \
        --PatchMatchStereo.max_image_size 2000 \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.gpu_index $GPU_INDEX \
        --PatchMatchStereo.depth_min 100 \
        --PatchMatchStereo.depth_max 400"
    N_GEO=$(ls "$DENSE_DIR/stereo/depth_maps/"*.geometric.bin 2>/dev/null | wc -l)
    log "Step 1: done — $N_GEO geometric depth maps."
fi

# ── Step 2: stereo_fusion ─────────────────────────────────────────────────────
FUSED_PLY="$DENSE_DIR/fused.ply"
if [[ "$SKIP_MVS" -eq 1 ]]; then
    log "Step 2: --skip-mvs set, skipping stereo_fusion."
    [[ -f "$FUSED_PLY" ]] || die "fused.ply not found at $FUSED_PLY (needed for step 4)"
elif [[ -f "$FUSED_PLY" ]]; then
    log "Step 2: fused.ply exists ($(du -sh "$FUSED_PLY" | cut -f1)), skipping stereo_fusion."
else
    log "Step 2: stereo_fusion ..."
    colmap_docker "colmap stereo_fusion \
        --workspace_path /workspace/dense/0 \
        --workspace_format COLMAP \
        --input_type photometric \
        --output_path /workspace/dense/0/fused.ply"
    log "Step 2: done — $(du -sh "$FUSED_PLY" | cut -f1)."
fi

# ── Step 3: poisson_mesher ────────────────────────────────────────────────────
MESH_PLY="$DENSE_DIR/meshed-poisson.ply"
if [[ "$SKIP_MVS" -eq 1 ]]; then
    log "Step 3: --skip-mvs set, skipping poisson_mesher."
    [[ -f "$MESH_PLY" ]] || die "meshed-poisson.ply not found at $MESH_PLY (needed for step 4)"
elif [[ -f "$MESH_PLY" ]]; then
    log "Step 3: meshed-poisson.ply exists ($(du -sh "$MESH_PLY" | cut -f1)), skipping poisson_mesher."
else
    log "Step 3: poisson_mesher ..."
    colmap_docker "colmap poisson_mesher \
        --input_path /workspace/dense/0/fused.ply \
        --output_path /workspace/dense/0/meshed-poisson.ply"
    log "Step 3: done — $(du -sh "$MESH_PLY" | cut -f1)."
fi

# ── Step 4: mesh → FuXian input ───────────────────────────────────────────────
# Faithful to the paper: Poisson-disk sample the COLMAP coarse mesh (built from
# the first-pass RGB via SfM + dense MVS) and attach per-point quality scores.
# (The earlier synthetic flat-plane bypass was removed — it discarded all building
#  geometry and broke the paper's coverage-driven planning. The dense MVS now works
#  because the v2 capture has ~72% overlap in both directions.)
SAMPLES_PLY="$REPO/data/${MAP_TAG}_samples.ply"
FUXIAN_MESH="$REPO/data/${MAP_TAG}_mesh.ply"
if [[ -f "$SAMPLES_PLY" && -f "$FUXIAN_MESH" ]]; then
    log "Step 4: FuXian PLYs exist, skipping mesh sampling."
else
    log "Step 4: Poisson-disk sampling of COLMAP mesh → FuXian inputs ..."
    mkdir -p "$REPO/data"
    python3 "$REPO/scripts/03_colmap_mesh_to_fuxian_input.py" \
        --mesh        "$MESH_PLY" \
        --dense-cloud "$FUSED_PLY" \
        --out-dir     "$REPO/data" \
        --tag         "$MAP_TAG" \
        --num-samples 4000 \
        --seed 42
    log "Step 4: done."
fi

# ── Step 5: Params_block9.h ───────────────────────────────────────────────────
PARAMS_H="$REPO/code/utils/Params_${MAP_TAG}.h"

# Read vertex count from PLY header (binary or ASCII)
NUM_SAMPLES=$(python3 - "$SAMPLES_PLY" <<'PYEOF'
import sys
with open(sys.argv[1], 'rb') as f:
    for line in f:
        if b'element vertex' in line:
            print(line.split()[-1].decode().strip())
            break
PYEOF
)
[[ -n "$NUM_SAMPLES" ]] || die "Could not read vertex count from $SAMPLES_PLY"
log "Step 5: $NUM_SAMPLES samples — writing $PARAMS_H"

cat > "$PARAMS_H" << HEOF
//
// Params for block9 (Matrix City / Small_City_LVL).
// Auto-generated by run_block9_mvs_to_plan.sh — do not edit by hand.
//
// Scene:  ~745 m x 390 m, building heights ~10-100 m
// Camera: 960x540, HFOV=45 deg, VFOV=26.23 deg
//
#ifndef FUXIAN_PARAMS_H
#define FUXIAN_PARAMS_H
#define M_PI 3.1415926

namespace Params{
    const std::string SAMPLE_FILE_PATH = "data/block9_samples.ply";
    const std::string MESH_FILE_PATH   = "data/block9_mesh.ply";

    // Wider range than Town01 to handle Matrix City skyscrapers (10-100 m)
    const float MIN_DISTANCE_BETWEEN_POINT_AND_VIEW = 25;
    const float MAX_DISTANCE_BETWEEN_POINT_AND_VIEW = 50;

    // 10 deg/step horizontal sampling (36 directions, same as Town01)
    const float PER_RADIAN = 360 / 36.0 / 180.0 * M_PI;

    const int POP_SIZE = 50;
    const int TOTAL_VIEW_NUMS = $NUM_SAMPLES;

    // 10 m voxel grid for viewpoint candidates
    const float VOXEL_SIZE = 10.0f;

    // 300 m: allows cross-block visibility in the 745 m wide scene
    const float MAX_D = 300;

    const float BEST_DISTANCE = 35;
    const float THETA_T = M_PI / 3.0f;
    const bool USE_PAPER_FOV_MARK = true;

    // Camera: 960x540, HFOV=45 deg
    // VFOV = 2*atan(tan(22.5 deg)*540/960) = 26.23 deg
    const double RESOLUTION_H = 960;
    const double RESOLUTION_V = 540;
    const double FOV_H        = 45.0;
    const double FOV_V        = 26.23;
}

#endif //FUXIAN_PARAMS_H
HEOF
log "Step 5: done."

# ── Step 6: FuXian path planning ──────────────────────────────────────────────
# Output file name mirrors main.cpp's suffix logic:
#   yan_two_stage      → block9_viewpoints.txt
#   binary_coverage    → block9_viewpoints_bc.txt
#   confidence_coverage→ block9_viewpoints_cwc.txt
VIEWPOINTS_FILE="$REPO/output/${MAP_TAG}/${MAP_TAG}_viewpoints${METHOD_SUFFIX}.txt"
log "Step 6: method=$METHOD  output=$(basename "$VIEWPOINTS_FILE")"
if [[ -f "$VIEWPOINTS_FILE" ]]; then
    log "Step 6: viewpoints file exists, skipping FuXian."
else
    log "Step 6: FuXian evolutionary path planning ..."
    (
        cd "$REPO"
        FUXIAN_MAP="$MAP_TAG" \
        FUXIAN_NAME="${MAP_TAG}_viewpoints" \
        FUXIAN_METHOD="$METHOD" \
            bash scripts/04_run_fuxian.sh
    )
    log "Step 6: done."
fi

[[ -f "$VIEWPOINTS_FILE" ]] || die "Viewpoints not found at $VIEWPOINTS_FILE"
VP_COUNT=$(wc -l < "$VIEWPOINTS_FILE")
log "  $VP_COUNT viewpoints."

# ── Step 7: viewpoints → CARLA plan ──────────────────────────────────────────
log "Step 7: viewpoints → CARLA second-pass plan [$METHOD] ..."
python3 "$REPO/scripts/05_fuxian_to_carla_plan.py" \
    --input  "$VIEWPOINTS_FILE" \
    --output "$PLAN_OUT" \
    --map    "$MAP_NAME" \
    --fov    45 \
    --width  960 \
    --height 540 \
    --one-group-per-waypoint

log ""
log "══════════════════════════════════════════════════════════"
log " Pipeline complete!  method=$METHOD"
log " CARLA second-pass plan → $PLAN_OUT"
log "══════════════════════════════════════════════════════════"
