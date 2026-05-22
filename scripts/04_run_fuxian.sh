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
    cmake ../code -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" 2>&1 | tail -5
    # tee the full make output to a log so failures aren't lost when grep filters
    # the on-screen output. PIPESTATUS[0] holds make's exit code (the trailing
    # `|| true` only masks grep's "no match" exit so set -e doesn't fire on it).
    make -j"$(nproc)" 2>&1 | tee /tmp/fuxian_build.log | grep -E "error|warning|Built|Linking" || true
    make_exit="${PIPESTATUS[0]}"
    if [ "$make_exit" -ne 0 ]; then
        echo "[fuxian] Build FAILED (make exit=$make_exit). See /tmp/fuxian_build.log"
        exit 1
    fi
)
echo "[fuxian] Build done."

echo "── Step 4c: Run FuXian path planning ────────────────────────────"
mkdir -p "output/${MAP_LOWER}" output/trajectory

echo "[fuxian] Input:"
echo "  mesh:    $(wc -c < "data/${MAP_LOWER}_mesh.ply") bytes"
echo "  samples: $(wc -l < "data/${MAP_LOWER}_samples.ply") lines"
echo ""

FUXIAN_TAG="${MAP_LOWER}" FUXIAN_NAME="${MAP_LOWER}_viewpoints" ./build/FuXian

echo ""
VIEWPOINTS_FILE="output/${MAP_LOWER}/${MAP_LOWER}_viewpoints.txt"
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
