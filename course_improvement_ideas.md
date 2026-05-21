# Course Improvement Plan

Baseline: Yan et al. 2021 paper-faithful reimplementation in this repo.

Course goal: keep the paper pipeline recognizable, then add one or more small, defensible terms/modifications and compare against the baseline.

Recommended framing:

> Diversity- and Energy-Aware UAV Viewpoint Planning for High-Quality Aerial 3D Reconstruction

Main proposal: combine Improvement 1 and Improvement 2. Improvement 3 is optional if we want a stronger quality-oriented variant.

---

## Baseline Recap

The paper selects viewpoints in two stages:

1. Local stage for low-quality samples `S_l`.
2. Global stage for all samples whose scanned count is below 2.
3. Path planning with ACO using translation and rotation costs.

Important baseline equations:

```text
Eq 8:  c(v_i, s_j) = w_v(v_i, s_j) · c_d(v_i, s_j) · c_o(v_i, s_j)

Eq 11: c_path(v_i, v_j) = 0.5 · c_t(v_i, v_j) + 0.5 · c_r(v_i, v_j)
```

The course improvements should modify these equations without replacing the whole method.

---

## Improvement 1: Coverage-Diversity Term for Viewpoint Selection

### Motivation

The paper's Eq 8 scores each candidate viewpoint independently for a sample. It does not directly penalize selecting viewpoints that are spatially redundant with already selected viewpoints, except for the coarse adjacent-voxel exclusion in the global stage.

In urban reconstruction, redundant viewpoints can waste images and flight time. A diversity term can encourage viewpoints to spread out while still respecting visibility, distance, and orientation confidence.

### Proposed Formula

Modify paper Eq 8:

```text
c'(v_i, s_j) = w_v · c_d · c_o · c_div(v_i)
```

where:

```text
c_div(v_i) = min(1, d_near(v_i, V_selected) / d_sep)
```

Definitions:

- `V_selected`: viewpoints already selected in local/global stages.
- `d_near(v_i, V_selected)`: Euclidean distance from candidate `v_i` to its nearest selected viewpoint.
- `d_sep`: desired minimum separation distance.

If no viewpoint has been selected yet:

```text
c_div(v_i) = 1
```

### Interpretation

- If a candidate is very close to an already selected viewpoint, `c_div` is small.
- If a candidate is far enough from selected viewpoints, `c_div = 1`, so there is no penalty.
- This keeps the paper's confidence score but discourages redundant viewpoints.

### Suggested Parameters

```text
USE_DIVERSITY_TERM = true
DIVERSITY_SEPARATION = 20.0 m
```

Possible tuning:

- `10 m`: weak diversity, close to baseline.
- `20 m`: reasonable starting point for current `dmin=25`, `dmax=35`.
- `30 m`: stronger spread, may reduce local-detail quality.

### Implementation Location

Primary file:

```text
code/utils/ea_utils.cpp
```

Add params:

```text
code/utils/Params.h
code/utils/Params_town01.h
code/utils/Params_town03.h
```

Implementation idea:

```cpp
float diversity = computeDiversity(candidatePositions[vIdx], finalSelections, candidatePositions);
ViewScore s(vIdx, i, cd * co * diversity);
```

Apply in:

- local stage scoring
- global stage scoring

### Expected Benefit

- Fewer redundant viewpoints.
- More spatially diverse coverage.
- Potentially shorter or smoother final ACO path if redundant viewpoints are avoided.

### Risk

- May hurt reconstruction if low-quality regions need nearby redundant views.
- Needs ablation against baseline because diversity can trade off detail capture.

---

## Improvement 2: Altitude/Energy Term for Path Planning

### Motivation

The paper's path cost uses translation and rotation:

```text
c_path = 0.5 · c_t + 0.5 · c_r
```

For UAVs, vertical movement is usually costly and can make flight less stable. The baseline translation term already includes 3D Euclidean distance, but it does not separately penalize altitude changes. A dedicated altitude term can reduce unnecessary climb/descent behavior.

### Proposed Formula

Modify paper Eq 11:

```text
c'_path(v_i, v_j) = α · c_t(v_i, v_j) + β · c_r(v_i, v_j) + γ · c_z(v_i, v_j)
```

where:

```text
c_z(v_i, v_j) = |z_i - z_j| / max_{a,b} |z_a - z_b|
```

Recommended weights:

```text
α = 0.45
β = 0.45
γ = 0.10
```

This keeps the paper's distance/rotation objective dominant while adding a mild altitude penalty.

### Implementation Location

Primary file:

```text
aco_tsp.py
```

Add CLI parameters:

```text
--w-translation 0.45
--w-rotation 0.45
--w-altitude 0.10
```

Current baseline equivalent:

```text
--w-translation 0.5 --w-rotation 0.5 --w-altitude 0.0
```

### Expected Benefit

- Less altitude oscillation.
- Lower vertical travel.
- Potentially smoother UAV motion.

### Risk

- May not improve reconstruction quality directly.
- If altitude penalty is too high, path may become less optimal for viewing geometry.

---

## Optional Improvement 3: Adaptive Local View Count for Low-Quality Samples

### Motivation

The paper local stage gives each low-quality sample in `S_l` exactly one dedicated local viewpoint. Very low-quality samples may benefit from more than one dedicated viewpoint instead of relying on the global stage for their second observation.

### Proposed Formula

Simple threshold version:

```text
n_local(s_i) =
  2, if g*(s_i) < τ_very_low
  1, otherwise for s_i ∈ S_l
```

Smooth version:

```text
n_local(s_i) = ceil(1 + λ · (1 - g*(s_i)))
```

Recommended simple version:

```text
τ_very_low = 0.25
```

### Implementation Location

Primary file:

```text
code/utils/ea_utils.cpp
```

Add params:

```text
USE_ADAPTIVE_LOCAL_VIEWS = true
VERY_LOW_QUALITY_THRESHOLD = 0.25
MAX_LOCAL_VIEWS_LOW_QUALITY = 2
```

### Expected Benefit

- More robust coverage of incomplete/low-quality regions.
- More likely to improve reconstruction completeness around difficult geometry.

### Risk

- More viewpoints and images.
- Harder to compare fairly unless image budget or path length is controlled.
- Teacher may view this as "just taking more pictures" unless we normalize by budget.

### Recommendation

Keep Improvement 3 as an optional ablation:

```text
baseline
baseline + diversity
baseline + diversity + altitude
baseline + diversity + altitude + adaptive local views
```

Only include it in the final method if it improves quality enough to justify extra images/path length.

---

## Recommended Experiment Variants

### Variant A: Paper Baseline

```text
USE_DIVERSITY_TERM = false
ACO altitude weight = 0
USE_ADAPTIVE_LOCAL_VIEWS = false
```

### Variant B: Diversity Only

```text
USE_DIVERSITY_TERM = true
ACO altitude weight = 0
USE_ADAPTIVE_LOCAL_VIEWS = false
```

Purpose:

- Isolate effect of diversity-aware viewpoint selection.

### Variant C: Diversity + Altitude/Energy

```text
USE_DIVERSITY_TERM = true
ACO altitude weight = 0.10
USE_ADAPTIVE_LOCAL_VIEWS = false
```

Purpose:

- Main recommended course method.
- Improves viewpoint redundancy and path smoothness while keeping image count close to baseline.

### Variant D: Diversity + Altitude + Adaptive Local Views

```text
USE_DIVERSITY_TERM = true
ACO altitude weight = 0.10
USE_ADAPTIVE_LOCAL_VIEWS = true
```

Purpose:

- Optional stronger reconstruction-quality variant.
- Use only if extra viewpoints/images are acceptable.

---

## Metrics to Report

### Planning Metrics

- Number of selected viewpoints.
- ACO path cost.
- Total path length.
- Mean / max yaw change.
- Mean / max pitch change.
- Total altitude change:

```text
Σ |z_i - z_{i+1}|
```

- Mean nearest-neighbor distance between viewpoints.

### Coverage Metrics

- Percentage of samples with scanned count >= 2.
- Percentage of low-quality samples covered >= 2.
- Average selected-view confidence for low-quality samples.

### Reconstruction Metrics

If final COLMAP reconstruction is available:

- final dense point count
- completeness against reference mesh/point cloud, if available
- reconstruction quality score distribution using the same Eq 1-5 evaluator
- qualitative comparison of hard regions

If no ground truth is available:

- compare final model density and coverage around low-quality first-pass regions
- visualize selected viewpoints and flight path

---

## Implementation Order

1. Implement Improvement 2 first in `aco_tsp.py`.
   - Lowest risk.
   - Easy to test with existing viewpoint text files.

2. Implement Improvement 1 in `ea_utils.cpp`.
   - Moderate risk.
   - Requires rebuilding FuXian.

3. Run planning-only smoke tests.
   - No CARLA/COLMAP.
   - Verify viewpoint count, ACO cost, altitude change, and output files.

4. Decide whether to implement Improvement 3.
   - Only add if we want a quality-oriented variant and can handle extra images.

---

## Final Recommendation

For the course project, use:

```text
Main method = baseline + coverage-diversity term + altitude/energy path term
```

Keep adaptive local view count as an optional ablation. It is promising, but it changes the image/path budget more aggressively, so it is less clean as the main claim.
