#!/usr/bin/env bash
# Run ACO on all viewpoint files that don't have ACO results yet.
set -euo pipefail
cd "$(dirname "$0")"

ANTS=20; ITERS=100; SEED=42

run_aco() {
    local input="$1" tag="$2" name="$3"
    local out_raw="output/${tag}/${name}_aco.txt"
    local out_smooth="output/${tag}/${name}_aco_smooth.txt"
    local out_smith="output/trajectory/${name}_smith_aco.txt"
    local out_smooth_smith="output/trajectory/${name}_smith_aco_smooth.txt"

    if [[ -f "$out_raw" ]]; then
        echo "SKIP (exists): $out_raw"
        return
    fi
    echo ""
    echo "========== ACO: $input =========="
    python3 aco_tsp.py \
        --input "$input" \
        --out-raw "$out_raw" \
        --out-smooth "$out_smooth" \
        --out-smith "$out_smith" \
        --out-smooth-smith "$out_smooth_smith" \
        --ants "$ANTS" --iters "$ITERS" --seed "$SEED"
}

# --- block9 (all three missing) ---
run_aco output/block9/block9_viewpoints.txt     block9 block9
run_aco output/block9/block9_viewpoints_bc.txt  block9 block9_bc
run_aco output/block9/block9_viewpoints_cwc.txt block9 block9_cwc

# --- town01: cwc hreq2_0 (default H_req=2.0) ---
run_aco output/town01/town01_viewpoints_hreq2_0_cwc.txt town01 town01_hreq2_0_cwc

# --- town03: all already exist, but run just in case ---
run_aco output/town03/town03_viewpoints.txt     town03 town03
run_aco output/town03/town03_viewpoints_bc.txt  town03 town03_bc
run_aco output/town03/town03_viewpoints_cwc.txt town03 town03_cwc

echo ""
echo "=== All done ==="
