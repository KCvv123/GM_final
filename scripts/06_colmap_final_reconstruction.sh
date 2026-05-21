#!/usr/bin/env bash
# =============================================================================
# 06_colmap_final_reconstruction.sh
#
# Step 9: Final reconstruction from second-pass images.
# Identical SfM + dense MVS pipeline as the first pass,
# but applied to the second-pass (FuXian path) images.
#
# Paper section: "Final Model"
#   Second-pass images → SfM → MVS → high-quality final dense model
#
# Usage:
#   ./scripts/06_colmap_final_reconstruction.sh \
#       --images /data/town01/second_pass/images \
#       --workspace /data/town01/colmap_second_pass
#
# Output:
#   <workspace>/dense/0/fused.ply          – final dense point cloud
#   <workspace>/dense/0/meshed-poisson.ply – final mesh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "================================================================"
echo " Step 9: Final reconstruction (second-pass images)"
echo "================================================================"
echo ""
echo "Delegating to 02_colmap_sfm_dense.sh with second-pass inputs..."
echo ""

exec "$SCRIPT_DIR/02_colmap_sfm_dense.sh" "$@"
