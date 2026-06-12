#!/usr/bin/env bash
# =============================================================================
# run_dense_only.sh
#
# COLMAP dense MVS on an EXISTING sparse model (skips feature/matching/SfM).
# Use when sparse/0 (triangulation) is already done and you only need the
# dense point cloud for paper-style accuracy/completeness evaluation.
#
# Per method:  image_undistorter -> patch_match_stereo -> stereo_fusion
# Output:      <workspace>/dense/0/fused.ply
#
# Runs COLMAP inside the SERVER's docker image (NOT vmlab-bin_colmap_38).
# =============================================================================
set -euo pipefail

# ---- EDIT THESE ------------------------------------------------------------
# Your server's COLMAP image (one of these — pick what `docker images` shows):
COLMAP_IMAGE="${COLMAP_IMAGE:-engine-colmap_38:latest}"   # or gennbv-bin_colmap_38:latest
GPU_INDEX="${GPU_INDEX:-0}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-3200}"   # >= your image width, so no downscale

# One entry per method:  <combined_images_dir>|<workspace_dir>
#   - combined_images_dir = round1 + round2 images for THAT method
#   - workspace_dir       = must ALREADY contain sparse/0 (triangulation output)
METHODS=(
  "/path/on/server/bc/images|/path/on/server/colmap_bc"
  "/path/on/server/cwc/images|/path/on/server/colmap_cwc"
  "/path/on/server/fuxian/images|/path/on/server/colmap_fuxian"
)
# ---------------------------------------------------------------------------

run_colmap() {   # $1=images_dir  $2=workspace  $3=command
  docker run --rm --gpus "device=$GPU_INDEX" \
    -v "$1":/images:ro \
    -v "$2":/workspace \
    "$COLMAP_IMAGE" \
    bash -c "$3"
}

for entry in "${METHODS[@]}"; do
  IMAGES="${entry%%|*}"
  WS="${entry##*|}"
  echo "==================================================================="
  echo " Dense MVS  →  $WS"
  echo "   images:   $IMAGES"
  echo "   image:    $COLMAP_IMAGE  (gpu $GPU_INDEX)"
  echo "==================================================================="

  if [[ ! -d "$WS/sparse/0" ]]; then
    echo "  !! $WS/sparse/0 not found — sparse model missing, skipping."
    continue
  fi
  mkdir -p "$WS/dense/0"

  echo "── 3a image_undistorter ──"
  run_colmap "$IMAGES" "$WS" "colmap image_undistorter \
    --image_path /images \
    --input_path /workspace/sparse/0 \
    --output_path /workspace/dense/0 \
    --output_type COLMAP \
    --max_image_size $MAX_IMAGE_SIZE"

  echo "── 3b patch_match_stereo (GPU, slow: ~20–40 min) ──"
  run_colmap "$IMAGES" "$WS" "colmap patch_match_stereo \
    --workspace_path /workspace/dense/0 \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.gpu_index 0"

  echo "── 3c stereo_fusion → fused.ply ──"
  run_colmap "$IMAGES" "$WS" "colmap stereo_fusion \
    --workspace_path /workspace/dense/0 \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path /workspace/dense/0/fused.ply"

  echo ">> DONE: $WS/dense/0/fused.ply"
  echo ""
done

echo "All methods done. Collect the three fused.ply for evaluation."
