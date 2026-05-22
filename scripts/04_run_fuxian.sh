#!/usr/bin/env bash
# =============================================================================
# 04_run_fuxian.sh
#
# Step 6: Run FuXian path planning algorithm (117-paper) on Town01 coarse model.
#
# Prerequisite: scripts/03_colmap_mesh_to_fuxian_input.py has been run,
#   producing data/town01_mesh.ply and data/town01_samples.ply
#
# Usage:
#   cd ~/safe_repos/117-Sampling-based-Path-Planning-for-High-quality-Aerial-3D-Reconstruction-of-Urban-Scenes
#   ./scripts/04_run_fuxian.sh
#
# Output:
#   output/town01/town01_viewpoints.txt  – viewpoints (x,y,z,dx,dy,dz per line)
#   output/trajectory/smithFormatPath*.txt – TSP-ordered path in Smith format
# =============================================================================

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

FUXIAN_MAP="${FUXIAN_MAP:-town01}"
MAP_LOWER=$(echo "$FUXIAN_MAP" | tr '[:upper:]' '[:lower:]')

# ---- Check inputs ----------------------------------------------------------
if [[ ! -f "data/${MAP_LOWER}_mesh.ply" ]]; then
    echo "[ERROR] data/${MAP_LOWER}_mesh.ply not found."
    echo "  Run: python3 scripts/03_colmap_mesh_to_fuxian_input.py --mesh <poisson.ply> --tag $MAP_LOWER"
    exit 1
fi
if [[ ! -f "data/${MAP_LOWER}_samples.ply" ]]; then
    echo "[ERROR] data/${MAP_LOWER}_samples.ply not found."
    exit 1
fi

echo "── Step 4a: Switch to ${MAP_LOWER} Params.h ───────────────────────────"
cp "code/utils/Params_${MAP_LOWER}.h" code/utils/Params.h
echo "[fuxian] Params.h updated."

echo "── Step 4b: Rebuild FuXian ───────────────────────────────────────"
mkdir -p build
(
    cd build
    cmake ../code -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" > /tmp/fuxian_cmake.log 2>&1 \
        || { tail -20 /tmp/fuxian_cmake.log; echo "[fuxian] cmake FAILED. See /tmp/fuxian_cmake.log"; exit 1; }
    tail -5 /tmp/fuxian_cmake.log
    # Run make WITHOUT piping so its exit status drives `set -e` directly.
    # An earlier version piped through `tee | grep ... || true` and read
    # PIPESTATUS[0], but `|| true` runs as a new pipeline and overwrites
    # PIPESTATUS, masking make failures (verified 2026-05-23 — would silently
    # re-use the stale ./build/FuXian binary).
    if ! make -j"$(nproc)" > /tmp/fuxian_build.log 2>&1; then
        echo "[fuxian] Build FAILED. Last lines of /tmp/fuxian_build.log:"
        tail -30 /tmp/fuxian_build.log
        exit 1
    fi
    grep -E "error|warning|Built|Linking" /tmp/fuxian_build.log || true
)
echo "[fuxian] Build done."

echo "── Step 4c: Run FuXian path planning ────────────────────────────"
mkdir -p "output/${MAP_LOWER}" output/trajectory

echo "[fuxian] Input:"
echo "  mesh:    $(wc -c < "data/${MAP_LOWER}_mesh.ply") bytes"
echo "  samples: $(wc -l < "data/${MAP_LOWER}_samples.ply") lines"
echo ""

FUXIAN_TAG="${MAP_LOWER}" FUXIAN_NAME="${MAP_LOWER}_viewpoints" ./build/FuXian

# main.cpp appends "_cwc" to FUXIAN_NAME when FUXIAN_METHOD=confidence_coverage,
# so the script must inspect the same suffixed file (or it would silently report
# stats for the previous baseline run that happened to be sitting in the dir).
SUFFIX=""
if [[ "${FUXIAN_METHOD:-}" == "confidence_coverage" ]]; then
    SUFFIX="_cwc"
fi

echo ""
VIEWPOINTS_FILE="output/${MAP_LOWER}/${MAP_LOWER}_viewpoints${SUFFIX}.txt"
if [[ ! -f "$VIEWPOINTS_FILE" ]]; then
    echo "[ERROR] Output not found: $VIEWPOINTS_FILE"
    exit 1
fi

VP_COUNT=$(wc -l < "$VIEWPOINTS_FILE")
echo "[fuxian] Generated $VP_COUNT viewpoints → $VIEWPOINTS_FILE"
echo ""
echo "── Step 4d: Preview first 5 viewpoints ─────────────────────────"
head -5 "$VIEWPOINTS_FILE"
echo ""
echo "[fuxian] Done.  Next step:"
echo "  python3 scripts/05_fuxian_to_carla_plan.py \\"
echo "      --input output/town01/town01_viewpoints.txt \\"
echo "      --output /data/town01/second_pass_plan.json"
