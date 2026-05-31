#!/usr/bin/env bash
# =============================================================================
# run_town01_paper_pipeline.sh
#
# Faithfully reproduces the 117-paper method on CARLA Town01.
#
# Paper: "Sampling-Based Path Planning for High-Quality Aerial 3D
#         Reconstruction of Urban Scenes" (Yan et al., Remote Sensing 2021)
#
# Pipeline:
#   Step 1  CARLA first-pass capture (nadir grid, 60m, 80%/70% overlap)
#   Step 2  COLMAP SfM  → sparse reconstruction
#   Step 3  COLMAP MVS  → dense point cloud  (patch_match_stereo + stereo_fusion)
#   Step 4  COLMAP Poisson mesher → coarse mesh
#   Step 5  Poisson-disk sampling on mesh → FuXian PLY input
#   Step 6  FuXian path planning → optimised viewpoints
#   Step 7  TSP-ordered second-pass CARLA capture plan (JSON)
#   Step 8  CARLA second-pass capture (FuXian path)
#   Step 9  COLMAP SfM + MVS on second-pass images → final dense model
#
# Requirements:
#   - gennbv-carla:0.9.16 Docker image  (for CARLA capture, steps 1 & 8)
#       Build: docker build -t gennbv-carla:0.9.16 \
#                  <gennbv_blackwall>/engine/envs/carla/
#   - vmlab-bin_colmap_38:latest Docker image  (for steps 2-4 & 9)
#   - Python 3 with numpy + scipy  (for step 5: scipy.spatial.cKDTree
#       is used for Eq 1-5 quality scoring; opencv-python required for
#       step 2 ORB key-image selection)
#   - C++ build tools (cmake + make)  for step 6 (FuXian)
#   - A running CARLA 0.9.16 server for steps 1 & 8 (launched by capturer)
#
# Usage:
#   cd ~/safe_repos/117-Sampling-based-Path-Planning-for-High-quality-Aerial-3D-Reconstruction-of-Urban-Scenes
#   ./run_town01_paper_pipeline.sh
#
#   # Resume from a specific step:
#   START_STEP=5 ./run_town01_paper_pipeline.sh
#
#   # Stop after a specific step (e.g. only generate the plan, no CARLA capture):
#   START_STEP=6 STOP_STEP=7 FUXIAN_METHOD=confidence_coverage FUXIAN_K=210 \
#       ./run_town01_paper_pipeline.sh
# =============================================================================

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

# ---- Config (override via env vars) ----------------------------------------
START_STEP="${START_STEP:-1}"
STOP_STEP="${STOP_STEP:-99}"   # inclusive upper bound; use e.g. 7 to stop after plan generation
MAP="${MAP:-Town01}"
MAP_LOWER=$(echo "$MAP" | tr '[:upper:]' '[:lower:]')
DATA_ROOT="${DATA_ROOT:-$HOME/safe_repos/117-paper-data-${MAP_LOWER}}"
GENNBV_ROOT="${GENNBV_ROOT:-$(dirname "$REPO")/gennbv_blackwall}"
CARLA_IMAGE="${CARLA_IMAGE:-gennbv-carla:0.9.16}"
GPU="${GPU:-0}"

# Per-method output suffix. Defined here (not inside Step 7) so that
# START_STEP=8 with FUXIAN_METHOD=confidence_coverage reads the correct
# CWC plan from Step 7's earlier run instead of a stale Yan plan, and so
# CARLA captures and the COLMAP final reconstruction land in their own
# dirs per planner instead of overwriting each other.
METHOD_SUFFIX=""
case "${FUXIAN_METHOD:-yan_two_stage}" in
    confidence_coverage) METHOD_SUFFIX="_cwc" ;;
    binary_coverage)     METHOD_SUFFIX="_bc"  ;;
esac

FIRST_PASS_DIR="$DATA_ROOT/first_pass"
COLMAP_FIRST_DIR="$DATA_ROOT/colmap_first_pass"
SECOND_PASS_DIR="$DATA_ROOT/second_pass${METHOD_SUFFIX}"
COLMAP_SECOND_DIR="$DATA_ROOT/colmap_second_pass${METHOD_SUFFIX}"
SECOND_PASS_PLAN="$DATA_ROOT/second_pass_plan${METHOD_SUFFIX}.json"
COLMAP_COMBINED_DIR="$DATA_ROOT/colmap_combined${METHOD_SUFFIX}"

print_step() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " Step $1: $2"
    echo "════════════════════════════════════════════════════════════════"
}

skip_step() {
    [[ $1 -lt $START_STEP ]] && echo "[skip] Step $1 (START_STEP=$START_STEP)" && return 0
    [[ $1 -gt $STOP_STEP ]]  && echo "[stop] Step $1 (STOP_STEP=$STOP_STEP)"   && return 0
    return 1
}

mkdir -p "$FIRST_PASS_DIR/images" "$COLMAP_FIRST_DIR" \
         "$SECOND_PASS_DIR/images" "$COLMAP_SECOND_DIR"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " 117-paper × CARLA $MAP — Full Paper Pipeline"
echo "════════════════════════════════════════════════════════════════"
echo "  REPO       = $REPO"
echo "  DATA_ROOT  = $DATA_ROOT"
echo "  START_STEP = $START_STEP"
echo ""

# =============================================================================
# STEP 1 — First-pass CARLA capture (nadir grid, Town01, 60m)
# =============================================================================
if ! skip_step 1; then
    print_step 1 "First-pass CARLA capture (nadir grid, 120m, 70/70% overlap)"

    CAPTURER="$GENNBV_ROOT/engine/core/nodes/carla/carla_capturer.py"
    if [[ ! -f "$CAPTURER" ]]; then
        echo "[ERROR] carla_capturer.py not found at: $CAPTURER"
        echo "  Set GENNBV_ROOT=<path to gennbv_blackwall>"
        exit 1
    fi

    # Minimal base config: global connection settings + capture_plan_path.
    # The plan JSON is mounted into the container and drives routes/groups.
    CAPTURE_CONFIG="$FIRST_PASS_DIR/capture_config.json"
    PLAN_JSON="$REPO/data/${MAP_LOWER}_first_pass_plan.json"
    python3 -c "
import json
plan = json.load(open('data/${MAP_LOWER}_first_pass_plan.json'))
cfg = dict(plan['globals'])
cfg['capture_plan_path'] = '/plan/first_pass_plan.json'
json.dump(cfg, open('$CAPTURE_CONFIG', 'w'), indent=2)
print('Capture config written.')
"

    echo "[step1] Running CARLA capture..."
    # CARLA server crashes with Signal 11 on shutdown (normal CARLA behaviour).
    # Ignore the non-zero exit code; verify success by image count instead.
    docker run --rm --network host \
        --gpus "device=$GPU" \
        --user "$(id -u):$(id -g)" \
        -v "$GENNBV_ROOT/engine/core/nodes/carla":/capturer:ro \
        -v "$FIRST_PASS_DIR":/data/output \
        -v "$CAPTURE_CONFIG":/config.json:ro \
        -v "$PLAN_JSON":/plan/first_pass_plan.json:ro \
        "$CARLA_IMAGE" \
        python3.10 /capturer/carla_capturer.py \
            --output_dir /data/output/images \
            --config_file /config.json || true

    IMG_COUNT=$(find "$FIRST_PASS_DIR/images/" -name "*.png" 2>/dev/null | wc -l)
    echo "[step1] Captured $IMG_COUNT images → $FIRST_PASS_DIR/images/"
    if [[ $IMG_COUNT -eq 0 ]]; then
        echo "[ERROR] No images captured. Check CARLA output above."
        exit 1
    fi
fi

# =============================================================================
# STEP 2-4 — COLMAP SfM + dense MVS + Poisson mesh (first pass)
# =============================================================================
if ! skip_step 2; then
    print_step "2-4" "COLMAP dense MVS + Poisson meshing (first pass, known poses)"

    # Paper Section 3.2.1 selects ORB key images before the coarse SfM/MVS
    # reconstruction. Keep the raw capture intact and write a key-image subset.
    RAW_FIRST_PASS_IMAGES="$FIRST_PASS_DIR/images"
    FIRST_PASS_IMAGES="$RAW_FIRST_PASS_IMAGES"
    POSES_JSON="$RAW_FIRST_PASS_IMAGES/poses.json"
    if [[ "${USE_ORB_KEY_IMAGES:-1}" != "0" ]]; then
        FIRST_PASS_KEY_IMAGES="$FIRST_PASS_DIR/key_images"
        rm -rf "$FIRST_PASS_KEY_IMAGES"
        KEY_ARGS=(
            --images "$RAW_FIRST_PASS_IMAGES"
            --output "$FIRST_PASS_KEY_IMAGES"
            --match-threshold "${ORB_MATCH_THRESHOLD:-450}"
            --min-features "${ORB_MIN_FEATURES:-150}"
            --max-features "${ORB_MAX_FEATURES:-4000}"
            --max-side "${ORB_MAX_SIDE:-1600}"
        )
        [[ -f "$POSES_JSON" ]] && KEY_ARGS+=(--poses "$POSES_JSON")
        python3 scripts/01_select_orb_key_images.py "${KEY_ARGS[@]}"
        FIRST_PASS_IMAGES="$FIRST_PASS_KEY_IMAGES"
        POSES_JSON="$FIRST_PASS_KEY_IMAGES/poses.json"
    fi

    # Use parent images/key_images dir (not subdirectory) so COLMAP names include
    # the group subdir (e.g. first_pass_nadir_120m/frame_0000.png),
    # matching what 02b_carla_poses_to_colmap.py writes into sparse model.
    IMG_COUNT=$(find "$FIRST_PASS_IMAGES" -name "*.png" | wc -l)
    if [[ $IMG_COUNT -eq 0 ]]; then
        echo "[ERROR] No PNG images found under $FIRST_PASS_IMAGES"
        exit 1
    fi
    echo "[step2-4] Using images dir: $FIRST_PASS_IMAGES ($IMG_COUNT images)"

    # CARLA gives exact camera poses in poses.json → skip SfM mapper entirely.
    # Convert poses.json → COLMAP sparse model, then run MVS directly.
    COLMAP_SPARSE="$COLMAP_FIRST_DIR/sparse/0"
    if [[ -f "$POSES_JSON" ]]; then
        echo "[step2-4] Using CARLA poses.json (bypassing SfM mapper)..."
        mkdir -p "$COLMAP_SPARSE"
        python3 scripts/02b_carla_poses_to_colmap.py \
            --poses  "$POSES_JSON" \
            --images "$FIRST_PASS_IMAGES" \
            --output "$COLMAP_SPARSE"

        ./scripts/02_colmap_sfm_dense.sh \
            --images "$FIRST_PASS_IMAGES" \
            --workspace "$COLMAP_FIRST_DIR" \
            --gpu "$GPU" \
            --skip-sfm
    else
        echo "[step2-4] No poses.json found, running full SfM..."
        ./scripts/02_colmap_sfm_dense.sh \
            --images "$FIRST_PASS_IMAGES" \
            --workspace "$COLMAP_FIRST_DIR" \
            --gpu "$GPU" \
            --matching exhaustive
    fi

    echo "[step2-4] Coarse mesh: $COLMAP_FIRST_DIR/dense/0/meshed-poisson.ply"
fi

# =============================================================================
# STEP 5 — Poisson-disk sampling → FuXian PLY input
# =============================================================================
if ! skip_step 5; then
    print_step 5 "Poisson-disk sampling on coarse mesh → FuXian input"

    COARSE_MESH="${COARSE_MESH_OVERRIDE:-$COLMAP_FIRST_DIR/dense/0/meshed-poisson.ply}"
    if [[ ! -f "$COARSE_MESH" ]]; then
        echo "[ERROR] Coarse mesh not found: $COARSE_MESH"
        exit 1
    fi
    echo "[step5] Using coarse mesh: $COARSE_MESH"

    DENSE_CLOUD="${DENSE_CLOUD_OVERRIDE:-$COLMAP_FIRST_DIR/dense/0/fused.ply}"
    python3 scripts/03_colmap_mesh_to_fuxian_input.py \
        --mesh    "$COARSE_MESH" \
        --dense-cloud "$DENSE_CLOUD" \
        --out-dir data/ \
        --tag "$MAP_LOWER" \
        --num-samples 2000

    echo "[step5] FuXian inputs: data/${MAP_LOWER}_mesh.ply, data/${MAP_LOWER}_samples.ply"
fi

# =============================================================================
# STEP 6 — FuXian path planning
# =============================================================================
if ! skip_step 6; then
    print_step 6 "FuXian sampling-based path planning"

    FUXIAN_MAP="$MAP_LOWER" ./scripts/04_run_fuxian.sh

    echo "[step6] Viewpoints: output/${MAP_LOWER}/${MAP_LOWER}_viewpoints.txt"
fi

# =============================================================================
# STEP 7 — Convert FuXian viewpoints → CARLA second-pass capture plan
# =============================================================================
if ! skip_step 7; then
    print_step 7 "ACO path planning + Bezier smoothing → CARLA second-pass plan"

    # METHOD_SUFFIX is computed once at the top of the script so START_STEP=8
    # also picks up the right plan; here we just reuse it.
    RAW_VIEWPOINTS="output/${MAP_LOWER}/${MAP_LOWER}_viewpoints${METHOD_SUFFIX}.txt"
    ACO_VIEWPOINTS="output/${MAP_LOWER}/${MAP_LOWER}_viewpoints${METHOD_SUFFIX}_aco.txt"
    ACO_SMOOTH_VIEWPOINTS="output/${MAP_LOWER}/${MAP_LOWER}_viewpoints${METHOD_SUFFIX}_aco_smooth.txt"

    python3 aco_tsp.py \
        --input "$RAW_VIEWPOINTS" \
        --out-raw "$ACO_VIEWPOINTS" \
        --out-smooth "$ACO_SMOOTH_VIEWPOINTS" \
        --out-smith "output/trajectory/${MAP_LOWER}${METHOD_SUFFIX}_smith_aco.txt" \
        --out-smooth-smith "output/trajectory/${MAP_LOWER}${METHOD_SUFFIX}_smith_aco_smooth.txt" \
        --ants "${ACO_ANTS:-20}" \
        --iters "${ACO_ITERS:-100}" \
        --alpha "${ACO_ALPHA:-1.0}" \
        --beta "${ACO_BETA:-2.0}" \
        --rho "${ACO_RHO:-0.1}" \
        --smooth-samples "${ACO_SMOOTH_SAMPLES:-4}" \
        --smooth-tension "${ACO_SMOOTH_TENSION:-0.35}" \
        --seed "${ACO_SEED:-42}"

    # Feed the ACO-ordered viewpoints (NOT the Bezier-smoothed expansion) into the
    # CARLA capture plan: CARLA teleports between waypoints rather than simulating
    # drone dynamics, so the 4-sample-per-segment Bezier inflation would turn K=210
    # selected viewpoints into 837 capture poses and break the budget-matched
    # comparison promised by proposal §3.1 ("exactly K second-pass camera poses").
    # The Bezier outputs are still produced above for trajectory visualisation.
    # SECOND_PASS_PLAN is already method-suffixed at the top of the script.
    python3 scripts/05_fuxian_to_carla_plan.py \
        --input  "$ACO_VIEWPOINTS" \
        --output "$SECOND_PASS_PLAN" \
        --map    "$MAP" \
        --preserve-order \
        --single-group-per-waypoint-orientation

    echo "[step7] Second-pass plan: $SECOND_PASS_PLAN"
fi

# =============================================================================
# STEP 8 — Second-pass CARLA capture (FuXian path)
# =============================================================================
if ! skip_step 8; then
    print_step 8 "Second-pass CARLA capture (FuXian planned path)"

    CAPTURER="$GENNBV_ROOT/engine/core/nodes/carla/carla_capturer.py"

    # Minimal base config for second pass — globals + plan path.
    # The second-pass plan JSON is mounted into the container.
    CAPTURE_CONFIG2="$SECOND_PASS_DIR/capture_config.json"
    python3 -c "
import json
plan = json.load(open('$SECOND_PASS_PLAN'))
cfg = dict(plan['globals'])
cfg['capture_plan_path'] = '/plan/second_pass_plan.json'
json.dump(cfg, open('$CAPTURE_CONFIG2', 'w'), indent=2)
print('Second-pass capture config written.')
"

    docker run --rm --network host \
        --gpus "device=$GPU" \
        --user "$(id -u):$(id -g)" \
        -v "$GENNBV_ROOT/engine/core/nodes/carla":/capturer:ro \
        -v "$SECOND_PASS_DIR":/data/output \
        -v "$CAPTURE_CONFIG2":/config.json:ro \
        -v "$SECOND_PASS_PLAN":/plan/second_pass_plan.json:ro \
        "$CARLA_IMAGE" \
        python3.10 /capturer/carla_capturer.py \
            --output_dir /data/output/images \
            --config_file /config.json || true

    IMG_COUNT=$(find "$SECOND_PASS_DIR/images/" -name "*.png" 2>/dev/null | wc -l)
    echo "[step8] Captured $IMG_COUNT images → $SECOND_PASS_DIR/images/"
    if [[ $IMG_COUNT -eq 0 ]]; then
        echo "[ERROR] No images captured. Check CARLA output above."
        exit 1
    fi
fi

# =============================================================================
# STEP 9 — Final COLMAP SfM + MVS on second-pass images
# =============================================================================
if ! skip_step 9; then
    print_step 9 "Final COLMAP SfM + dense MVS (second-pass images)"

    # Second-pass images span multiple subdirs (one per pitch angle),
    # so use the parent images/ directory, not a single subdir.
    SECOND_PASS_IMAGES="$SECOND_PASS_DIR/images"
    IMG_COUNT=$(find "$SECOND_PASS_IMAGES" -name "*.png" | wc -l)
    if [[ $IMG_COUNT -eq 0 ]]; then
        echo "[ERROR] No PNG images found under $SECOND_PASS_IMAGES"
        exit 1
    fi
    echo "[step9] Using images dir: $SECOND_PASS_IMAGES ($IMG_COUNT images across subdirs)"

    POSES_JSON2="$SECOND_PASS_DIR/images/poses.json"
    COLMAP_SPARSE2="$COLMAP_SECOND_DIR/sparse/0"
    if [[ -f "$POSES_JSON2" ]]; then
        echo "[step9] Using CARLA poses.json (bypassing SfM mapper)..."
        mkdir -p "$COLMAP_SPARSE2"
        python3 scripts/02b_carla_poses_to_colmap.py \
            --poses  "$POSES_JSON2" \
            --images "$SECOND_PASS_IMAGES" \
            --output "$COLMAP_SPARSE2"

        ./scripts/06_colmap_final_reconstruction.sh \
            --images "$SECOND_PASS_IMAGES" \
            --workspace "$COLMAP_SECOND_DIR" \
            --gpu "$GPU" \
            --skip-sfm
    else
        echo "[step9] No poses.json found, running full SfM..."
        ./scripts/06_colmap_final_reconstruction.sh \
            --images "$SECOND_PASS_IMAGES" \
            --workspace "$COLMAP_SECOND_DIR" \
            --gpu "$GPU" \
            --matching exhaustive
    fi

    echo "[step9] Final dense model: $COLMAP_SECOND_DIR/dense/0/fused.ply"
    echo "[step9] Final mesh:        $COLMAP_SECOND_DIR/dense/0/meshed-poisson.ply"
fi

# =============================================================================
# STEP 10 — Combined reconstruction (all passes)
# =============================================================================
# Set EXTRA_IMAGE_DIRS to space-separated list of extra capture directories
# (each must contain images/ subdir and optionally images/poses.json).
# Example: EXTRA_IMAGE_DIRS="$HOME/data/second_pass_round2" START_STEP=10 ./run_town01_paper_pipeline.sh
if ! skip_step 10; then
    print_step 10 "Combined COLMAP reconstruction (all passes)"

    COMBINED_IMAGES="$DATA_ROOT/combined_images"
    rm -rf "$COMBINED_IMAGES"
    mkdir -p "$COMBINED_IMAGES"

    # ---- Helper: hardlink a directory of PNGs into combined_images/<label>/ ----
    hardlink_pass() {
        local SRC_IMG_DIR="$1"  # directory containing PNGs (possibly in subdirs)
        local LABEL="$2"        # e.g. first_pass, second_pass, round3
        find "$SRC_IMG_DIR" -name "*.png" | while read src; do
            rel=$(realpath --relative-to="$SRC_IMG_DIR" "$src")
            subdir=$(dirname "$rel")
            mkdir -p "$COMBINED_IMAGES/$LABEL/$subdir"
            ln -f "$src" "$COMBINED_IMAGES/$LABEL/$rel" 2>/dev/null \
                || cp "$src" "$COMBINED_IMAGES/$LABEL/$rel"
        done
        local CNT
        CNT=$(find "$COMBINED_IMAGES/$LABEL" -name "*.png" 2>/dev/null | wc -l)
        echo "[step10] $LABEL: $CNT images" >&2
        echo "$CNT"
    }

    TOTAL=0

    echo "[step10] Hardlinking first-pass images..."
    N=$(hardlink_pass "$FIRST_PASS_DIR/images" "first_pass")
    TOTAL=$((TOTAL + N))

    echo "[step10] Hardlinking second-pass images..."
    N=$(hardlink_pass "$SECOND_PASS_DIR/images" "second_pass")
    TOTAL=$((TOTAL + N))

    # Extra passes (previous rounds, third pass, etc.)
    EXTRA_NUM=0
    for EXTRA_DIR in ${EXTRA_IMAGE_DIRS:-}; do
        EXTRA_NUM=$((EXTRA_NUM + 1))
        EXTRA_LABEL="extra_pass_${EXTRA_NUM}"
        if [[ -d "$EXTRA_DIR/images" ]]; then
            EXTRA_SRC="$EXTRA_DIR/images"
        elif [[ -d "$EXTRA_DIR" ]]; then
            EXTRA_SRC="$EXTRA_DIR"
        else
            echo "[step10] WARNING: $EXTRA_DIR not found, skipping"
            continue
        fi
        echo "[step10] Hardlinking $EXTRA_LABEL from $EXTRA_SRC..."
        N=$(hardlink_pass "$EXTRA_SRC" "$EXTRA_LABEL")
        TOTAL=$((TOTAL + N))
    done

    echo "[step10] Total combined: $TOTAL images"

    # ---- Merge all poses.json files ----
    # Collect all (label, poses_path) pairs
    ALL_POSES_ARGS=""
    [[ -f "$FIRST_PASS_DIR/images/poses.json" ]] && \
        ALL_POSES_ARGS="$ALL_POSES_ARGS first_pass $FIRST_PASS_DIR/images/poses.json"
    [[ -f "$SECOND_PASS_DIR/images/poses.json" ]] && \
        ALL_POSES_ARGS="$ALL_POSES_ARGS second_pass $SECOND_PASS_DIR/images/poses.json"
    EXTRA_NUM=0
    for EXTRA_DIR in ${EXTRA_IMAGE_DIRS:-}; do
        EXTRA_NUM=$((EXTRA_NUM + 1))
        EXTRA_LABEL="extra_pass_${EXTRA_NUM}"
        EXTRA_POSES=""
        [[ -f "$EXTRA_DIR/images/poses.json" ]] && EXTRA_POSES="$EXTRA_DIR/images/poses.json"
        [[ -z "$EXTRA_POSES" && -f "$EXTRA_DIR/poses.json" ]] && EXTRA_POSES="$EXTRA_DIR/poses.json"
        [[ -n "$EXTRA_POSES" ]] && ALL_POSES_ARGS="$ALL_POSES_ARGS $EXTRA_LABEL $EXTRA_POSES"
    done

    if [[ -n "$ALL_POSES_ARGS" ]]; then
        echo "[step10] Merging poses.json (known camera poses)..."
        python3 -c "
import json, copy, sys

args = '$ALL_POSES_ARGS'.split()
pairs = [(args[i], args[i+1]) for i in range(0, len(args), 2)]

base = None
merged_frames = []

for label, path in pairs:
    with open(path) as f:
        data = json.load(f)
    if base is None:
        base = copy.deepcopy(data)
    for fr in data['frames']:
        f = copy.deepcopy(fr)
        parts = f['image_path'].split('/images/')
        rel = parts[-1] if len(parts) > 1 else f['image_path'].split('/')[-1]
        f['image_path'] = label + '/' + rel
        f['frame_idx'] = len(merged_frames)
        merged_frames.append(f)
    print(f'  {label}: {len(data[\"frames\"])} frames from {path}')

base['frames'] = merged_frames
with open('$COMBINED_IMAGES/poses.json', 'w') as f:
    json.dump(base, f, indent=2)
print(f'[step10] Total merged: {len(merged_frames)} frames')
"
        # Build sparse model from merged poses
        COLMAP_COMBINED_SPARSE="$COLMAP_COMBINED_DIR/sparse/0"
        mkdir -p "$COLMAP_COMBINED_SPARSE"
        python3 scripts/02b_carla_poses_to_colmap.py \
            --poses  "$COMBINED_IMAGES/poses.json" \
            --images "$COMBINED_IMAGES" \
            --output "$COLMAP_COMBINED_SPARSE"

        ./scripts/02_colmap_sfm_dense.sh \
            --images "$COMBINED_IMAGES" \
            --workspace "$COLMAP_COMBINED_DIR" \
            --gpu "$GPU" \
            --skip-sfm
    else
        echo "[step10] No poses.json found, running full SfM..."
        mkdir -p "$COLMAP_COMBINED_DIR"
        ./scripts/02_colmap_sfm_dense.sh \
            --images "$COMBINED_IMAGES" \
            --workspace "$COLMAP_COMBINED_DIR" \
            --gpu "$GPU" \
            --matching exhaustive
    fi

    echo "[step10] Combined dense cloud: $COLMAP_COMBINED_DIR/dense/0/fused.ply"
    echo "[step10] Combined mesh:        $COLMAP_COMBINED_DIR/dense/0/meshed-poisson.ply"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo " Pipeline complete!"
echo ""
echo "  First-pass images:   $FIRST_PASS_DIR/images/"
echo "  Second-pass images:  $SECOND_PASS_DIR/images/"
echo "  Second-pass only:    $COLMAP_SECOND_DIR/dense/0/fused.ply"
echo "  Combined dense cloud: $COLMAP_COMBINED_DIR/dense/0/fused.ply"
echo "  Combined mesh:        $COLMAP_COMBINED_DIR/dense/0/meshed-poisson.ply"
echo "================================================================"
