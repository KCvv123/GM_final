#!/usr/bin/env bash
# =============================================================================
# 02_colmap_sfm_dense.sh
#
# Step 2-4: COLMAP SfM + dense MVS + Poisson meshing
# Faithfully follows the paper's coarse model pipeline:
#   SfM → sparse point cloud
#   patch_match_stereo → per-image depth maps
#   stereo_fusion → fused dense point cloud
#   poisson_mesher → coarse mesh  (input to 117-paper FuXian)
#
# Usage:
#   ./scripts/02_colmap_sfm_dense.sh \
#       --images /data/town01/first_pass/images \
#       --workspace /data/town01/colmap_first_pass
#
# Output:
#   <workspace>/sparse/0/          – SfM sparse model
#   <workspace>/dense/0/fused.ply  – dense point cloud
#   <workspace>/dense/0/meshed-poisson.ply  – coarse Poisson mesh  ← used by FuXian
#
# Requirements:
#   Docker image vmlab-bin_colmap_38:latest with COLMAP 3.8
# =============================================================================

set -euo pipefail

# ---- Parse args ------------------------------------------------------------
IMAGES_DIR=""
WORKSPACE=""
GPU_INDEX="${GPU_INDEX:-0}"
NUM_THREADS="${NUM_THREADS:-8}"
MATCHING_MODE="${MATCHING_MODE:-exhaustive}"   # exhaustive | sequential
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-3200}"       # downsample for stereo
COLMAP_IMAGE="vmlab-bin_colmap_38:latest"
SKIP_SFM=0   # set to 1 to skip feature extraction/matching/mapper (known poses)

usage() {
    echo "Usage: $0 --images <dir> --workspace <dir> [--skip-sfm]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --images)    IMAGES_DIR="$(realpath "$2")"; shift 2 ;;
        --workspace) WORKSPACE="$(realpath "$2")";  shift 2 ;;
        --gpu)       GPU_INDEX="$2"; shift 2 ;;
        --matching)  MATCHING_MODE="$2"; shift 2 ;;
        --skip-sfm)  SKIP_SFM=1; shift ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

[[ -z "$IMAGES_DIR" || -z "$WORKSPACE" ]] && usage

mkdir -p "$WORKSPACE/sparse" "$WORKSPACE/dense/0"

DB="$WORKSPACE/database.db"

echo "================================================================"
echo " COLMAP SfM + Dense MVS Pipeline"
echo "================================================================"
echo "  images:    $IMAGES_DIR"
echo "  workspace: $WORKSPACE"
echo "  matching:  $MATCHING_MODE"
echo ""

# Generate image list (PNG/JPG only) so COLMAP skips non-image files
# (e.g. poses.json, capture_metadata.csv) that live alongside images.
IMAGE_LIST="$WORKSPACE/image_list.txt"
(cd "$IMAGES_DIR" && find -L . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sed 's|^\./||' | sort) > "$IMAGE_LIST"
IMAGE_LIST_COUNT=$(wc -l < "$IMAGE_LIST")
echo "[colmap] Image list: $IMAGE_LIST_COUNT images → $IMAGE_LIST"

# Helper: run COLMAP inside Docker
run_colmap() {
    local CMD="$1"
    echo "[colmap] $CMD" | head -c 120
    echo ""
    docker run --rm --gpus "device=$GPU_INDEX" \
        -v "$IMAGES_DIR":/images:ro \
        -v "$WORKSPACE":/workspace \
        "$COLMAP_IMAGE" \
        bash -c "$CMD"
}

if [[ $SKIP_SFM -eq 1 ]]; then
    echo "── Steps 2a-2b: Feature extraction + matching (poses are known) ──"
    # Extract features and match them — we need co-visibility for MVS.
    run_colmap "colmap feature_extractor \
        --database_path /workspace/database.db \
        --image_path /images \
        --image_list_path /workspace/image_list.txt \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model PINHOLE \
        --SiftExtraction.max_num_features 8192"

    run_colmap "colmap exhaustive_matcher \
        --database_path /workspace/database.db"

    echo "── Step 2c: point_triangulator (known poses + matched features → 3D points) ──"
    run_colmap "colmap point_triangulator \
        --database_path /workspace/database.db \
        --image_path /images \
        --input_path /workspace/sparse/0 \
        --output_path /workspace/sparse/0"

    run_colmap "colmap model_analyzer --path /workspace/sparse/0"
else
    # ---- Step 2a: Feature extraction -----------------------------------------
    echo "── Step 2a: Feature extraction ──────────────────────────────────"
    run_colmap "colmap feature_extractor \
        --database_path /workspace/database.db \
        --image_path /images \
        --image_list_path /workspace/image_list.txt \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model PINHOLE \
        --SiftExtraction.max_num_features 8192"

    # ---- Step 2b: Feature matching -------------------------------------------
    echo "── Step 2b: Feature matching ($MATCHING_MODE) ───────────────────"
    if [[ "$MATCHING_MODE" == "sequential" ]]; then
        run_colmap "colmap sequential_matcher \
            --database_path /workspace/database.db \
            --SequentialMatching.loop_detection 1"
    else
        run_colmap "colmap exhaustive_matcher \
            --database_path /workspace/database.db"
    fi

    # ---- Step 2c: Incremental SfM (mapper) -----------------------------------
    echo "── Step 2c: SfM mapper ──────────────────────────────────────────"
    run_colmap "colmap mapper \
        --database_path /workspace/database.db \
        --image_path /images \
        --output_path /workspace/sparse"

    # COLMAP mapper may create multiple models (0, 1, ...).  Pick the largest.
    LARGEST_MODEL=$(docker run --rm \
        -v "$WORKSPACE":/workspace \
        "$COLMAP_IMAGE" \
        bash -c "ls /workspace/sparse/ | while read d; do \
            n=\$(colmap model_analyzer --path /workspace/sparse/\$d 2>/dev/null | grep 'Images' | awk '{print \$2}'); \
            echo \"\$n \$d\"; done | sort -rn | head -1 | awk '{print \$2}'" 2>/dev/null || echo "0")

    echo "[colmap] Largest sparse model: $LARGEST_MODEL"

    if [[ "$LARGEST_MODEL" != "0" && -d "$WORKSPACE/sparse/$LARGEST_MODEL" ]]; then
        cp -r "$WORKSPACE/sparse/$LARGEST_MODEL"/* "$WORKSPACE/sparse/0/"
    fi

    run_colmap "colmap model_analyzer --path /workspace/sparse/0"
fi

# ---- Step 3: Dense MVS – image undistortion --------------------------------
echo "── Step 3a: Image undistortion ──────────────────────────────────"
run_colmap "colmap image_undistorter \
    --image_path /images \
    --input_path /workspace/sparse/0 \
    --output_path /workspace/dense/0 \
    --output_type COLMAP \
    --max_image_size $MAX_IMAGE_SIZE"

# ---- Step 3b: Dense stereo (depth maps) ------------------------------------
echo "── Step 3b: patch_match_stereo ──────────────────────────────────"
run_colmap "colmap patch_match_stereo \
    --workspace_path /workspace/dense/0 \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.gpu_index 0"

# ---- Step 3c: Stereo fusion → dense point cloud ----------------------------
echo "── Step 3c: stereo_fusion ───────────────────────────────────────"
run_colmap "colmap stereo_fusion \
    --workspace_path /workspace/dense/0 \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path /workspace/dense/0/fused.ply"

# ---- Step 4: Poisson meshing ------------------------------------------------
echo "── Step 4: Poisson mesher (coarse mesh for FuXian) ──────────────"
run_colmap "colmap poisson_mesher \
    --input_path /workspace/dense/0/fused.ply \
    --output_path /workspace/dense/0/meshed-poisson.ply"

echo ""
echo "================================================================"
echo " Done!"
echo "  Sparse:      $WORKSPACE/sparse/0/"
echo "  Dense cloud: $WORKSPACE/dense/0/fused.ply"
echo "  Coarse mesh: $WORKSPACE/dense/0/meshed-poisson.ply  ← FuXian input"
echo "================================================================"
