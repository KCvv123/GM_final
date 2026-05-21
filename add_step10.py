#!/usr/bin/env python3
"""Rewrite step 10 in run_town01_paper_pipeline.sh to support EXTRA_IMAGE_DIRS."""
import re

SCRIPT = "run_town01_paper_pipeline.sh"

with open(SCRIPT) as f:
    text = f.read()

# Remove existing step 10 block (from "# STEP 10" to just before "# Summary")
text = re.sub(
    r'\n*# =+\n# STEP 10.*?(?=# =+\n# Summary)',
    '\n',
    text,
    flags=re.DOTALL,
)

STEP10 = r'''
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
        echo "[step10] $LABEL: $CNT images"
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

'''

# Insert before Summary
text = text.replace(
    '# =============================================================================\n# Summary',
    STEP10 + '# =============================================================================\n# Summary',
)

with open(SCRIPT, 'w') as f:
    f.write(text)

print("Done - step 10 rewritten with EXTRA_IMAGE_DIRS support")
