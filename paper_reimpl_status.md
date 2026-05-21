# 117-Paper Reimplementation Status

Yan et al. 2021, "Sampling-Based Path Planning for High-Quality Aerial 3D Reconstruction of Urban Scenes" (`remotesensing-13-00989.pdf`).

**Dual purpose:** (1) faithful baseline used as a comparison in the user's own research; (2) base for a course assignment that improves on the paper. Keep `main` paper-faithful; course improvements branch off.

Last updated: 2026-05-06.

---

## Pipeline overview

`run_town01_paper_pipeline.sh` orchestrates the full Town01 run. Stage-wise:

| Stage | Script / code | Paper § |
|---|---|---|
| CARLA poses → COLMAP | `scripts/02b_carla_poses_to_colmap.py` | input prep |
| COLMAP SfM + dense + Poisson mesh | `scripts/02_colmap_sfm_dense.sh` | 3.1 |
| Mesh → samples + quality scores → `.ply` for FuXian | `scripts/03_colmap_mesh_to_fuxian_input.py` | 3.2.2 (sampling), 3.2.3 (Eq 1–5) |
| FuXian C++ binary: candidates + viewpoint selection | `code/main.cpp` → `code/utils/ea_utils.cpp` | 3.3 |
| FuXian output → CARLA flight plan | `scripts/05_fuxian_to_carla_plan.py` | 3.4 (path) |
| Final reconstruction with new images | `scripts/06_colmap_final_reconstruction.sh` | — |

---

## Alignment status

### ✅ Aligned with paper

| Paper § | Implementation | Notes |
|---|---|---|
| 3.2.2 OBB (PCA on normals, vertical-axis constrained) | `code/entity/map.cpp` `initOBB` | |
| 3.2.2 Poisson mesh + Poisson-disk surface sampling | COLMAP scripts + `scripts/03_*` `poisson_disk_sample_mesh` | |
| 3.3.1 Voxel-grid candidates with `dmin/dmax` shell | `ea_utils.cpp` `generateVoxelCandidates` | `VOXEL_SIZE=10`, `dmin=25`, `dmax=35`. Uses nearest-sample distance as object distance — engineering approximation; should be sanity-checked once Poisson-disk radius `R` is fixed (B1). |
| 3.3.2 Eq 6 c_d (distance confidence) | `ea_utils.cpp` local/global scoring blocks | |
| 3.3.2 Eq 7 c_o (orientation confidence) | `ea_utils.cpp` local/global scoring blocks | includes `θ_t` denominator and hard threshold; `θ_t=60°` is an assumption |
| 3.3.2 Eq 8 confidence = w_v · c_d · c_o | `ea_utils.cpp` local/global scoring blocks | |
| 3.3.2 Local stage: S_l only, 1 best unchosen viewpoint each | `ea_utils.cpp` `selectViewPointByScore` (local block) | high-quality samples skip local |
| 3.3.2 Global stage: all samples scanned ≥ 2, adjacent voxel exclusion | `ea_utils.cpp` `selectViewPointByScore` (global block) + `isAdjacentToAnySelected` | 6-connected, `VOXEL_SIZE × 1.1` slack. Scanned mark is switchable: paper FOV-radius mark by default; ray-visibility variant via `USE_PAPER_FOV_MARK=false`. |
| 3.3.2 Orientation bound to trigger sample s_j | `ViewSelection{viewIdx, triggerSample}` + `selectionsToViewPoints` | dedupe by voxel (paper says "best **unchosen**") |
| Embree ray-only visibility (w_v) + no-look-up rule | `score.cpp` `isVisibleRayOnly` | samples above candidate viewpoint are invisible by default |
| No EA refinement in main pipeline | `main.cpp` calls `initPopulationWithContext` → `selectionsToViewPoints` only | `runEA` removed; paper has no EA stage |

### ❌ Pending for paper-faithful baseline

#### B1. Quality scoring formulas (Eq 1–5) wrong  *(approximation expected)* ✅ fixed
- **Where:** `scripts/03_colmap_mesh_to_fuxian_input.py` `compute_quality_scores`
- **Paper:**
  - Eq 1 `g_c(s_i) = w_i · ∇(s_i) · n_i` — Poisson gradient × normal × density weight
  - Eq 2 `g_s(s_i) = Σ w_pj · exp{-(1 − N_pj · n_i)² / (1 − cos θ_t)²}` — neighbourhood smoothness
  - Density weight `w_i = (1 − (|N_i| − |N_max|) / |N_max|)²`
  - Eq 4 product, Eq 5 normalize to [0, 1]
- **Now:** implemented A3 approximation for Eq 1 plus verbatim Eq 2/3/4/5 structure:
  - `∇(s_i) ≈ Σ w(‖s_i − p_j‖) · n_pj_input`, normalized before the dot product with `n_i`
  - `w_i = (1 − (|N_i| − |Nmax|) / |Nmax|)^2`
  - `w_pj = exp{-‖s_i - p_j‖₂ / R}` (Eq 3) — **fixed 2026-05-22**: previous code used squared distance `d² / R`, paper formula is Euclidean (non-squared). Effective neighbourhood radius was ~½ of paper's at R=3.0.
  - `g_s = Σ w_pj · exp{-(1 − N_pj · n_i)^2 / (1 − cos θ_t)^2}`
  - `g = g_c · g_s`, then Eq 5 min-max normalization to [0, 1]
- **New CLI parameters:** `--quality-radius` controls neighbourhood radius `R`; `--theta-t-deg` controls `θ_t` and defaults to 60°.
- **⚠️ Explicit caveat:** A3 is a *faithful approximation* of `∇(s_i)`, **not** bit-level reproduction. Paper's Poisson gradient is unavailable without patching PoissonRecon source; A3 captures the documented intent (gradient is a smoothed input vector field) and produces values in the right scale and direction, but a strict-reproducibility argument cannot rest on this term.
- **Verification:** synthetic point-cloud check passed 2026-05-06 without running COLMAP.

#### B2. "No-look-up" filter not enforced ✅ fixed
- **Where:** `code/utils/score.cpp` `isVisibleRayOnly` (around `:203`)
- **Paper 3.3.2(i):** "the sample points above the candidate viewpoint are invisible by default" (UAV does not look upward).
- **Now:** `isVisibleRayOnly` returns false before tracing when `sample.z > viewpoint.z + epsilon`.
- **Fix:** implemented 2026-05-06.
- **Risk:** without it, low-quality samples on building tops can be "covered" by viewpoints beneath them — physically wrong.
- **Cost:** one-line guard; cheap.

#### B3. Eq 7 c_o missing θ_t denominator ✅ fixed
- **Where:** `code/utils/ea_utils.cpp:161` (local stage) and `code/utils/ea_utils.cpp:218` (global stage)
- **Paper Eq 7:** `c_o(v_i^c, s_j) = exp{-(1 − n_s · d)² / (1 − cos θ_t)²}` — and the angle "should be within a threshold θ_t".
- **Now:** `Params::THETA_T = 60°`, pair candidates are hard-rejected when `n_s · d < cos(θ_t)`, and `co = exp(-pow(1 - nd, 2) / pow(1 - cos(θ_t), 2))`.
- **Fix:** implemented 2026-05-06.
- **Risk:** changes viewpoint ranking. Moderate impact — the previous formula was too broad / too lenient for any assumed `θ_t < 90°`.

#### B4. Decide global "scanned" mark mechanism ✅ fixed
- **Where:** `code/utils/ea_utils.cpp` global stage, coverage init around `:186`.
- **Paper 3.3.2 (global):** for each local viewpoint with trigger sample s_j, mark all samples within radius `\|d(v_l, s_j)\| · sin(θ_fov/2)` of s_j as "scanned".
- **Now:** `Params::USE_PAPER_FOV_MARK` controls the mode.
  - `true` (default): paper-literal FOV-radius mark using `\|d(v_l, s_j)\| · sin(θ_fov/2)` around the trigger sample.
  - `false`: ray-visibility variant using `viewPointVisibilitySet` from Embree visibility.
- **Fix:** implemented 2026-05-06. This allows running both the strict reimplementation and the visibility-refined variant.

#### B5. Path planning — wiring + Bezier missing ✅ fixed
- **Where:** `aco_tsp.py`; `run_town01_paper_pipeline.sh` Step 7; `scripts/05_fuxian_to_carla_plan.py`
- **Paper 3.4:**
  - Eq 9 translation cost `c_t = (b − b_min)/(b_max − b_min)`
  - Eq 10 rotation cost `c_r = ½(1 − o_i · o_j / ‖o_i‖‖o_j‖)`
  - Eq 11 `c = 0.5·c_t + 0.5·c_r`
  - Solve TSP via ACO ([34], ant-cycle [35]); smooth with cubic Bezier.
- **Now:**
  1. `aco_tsp.py` builds Eq 9/10/11 cost matrix, runs ACO, and writes ACO-ordered raw viewpoints.
  2. `aco_tsp.py` also writes a cubic-Bezier-smoothed path (`--out-smooth`, `--out-smooth-smith`).
  3. `run_town01_paper_pipeline.sh` Step 7 calls `aco_tsp.py` before CARLA plan conversion.
  4. `scripts/05_fuxian_to_carla_plan.py` supports `--preserve-order` and `--one-group-per-waypoint`, so it does not undo ACO/Bezier ordering and can preserve per-viewpoint z under the existing CARLA plan schema.
- **Open-path decision:** implemented as open Hamiltonian ordering because the paper does not state UAV return-to-start and the CARLA capture mission is naturally open. This is documented as an assumption.
- **Verification:** handcrafted 5-viewpoint check passed 2026-05-06; ACO reduced cost, Bezier output had expected line count, CARLA JSON was generated in preserve-order mode.

#### B6. Normal/location clusters (Section 3.2.2)
- **Where:** would live in `code/entity/map.cpp` (after `initOBB`)
- **Paper 3.2.2:** build clusters `C = {C₁,…,Cₖ}` by surface position and normal direction; both local and global selection are then run "in parallel for each sample point cluster".
- **Now:** all selection iterates flat over `points`.
- **Why this is *not* purely performance:** if cluster-local greedy uses an independent `selected` set per cluster, the union ≠ flat-greedy result (the "best unchosen" lock differs). Even with a shared selected set, ordering effects can change which viewpoint a particular sample is bound to. Practical impact is likely small but not provably zero.
- **Recommendation:** defer; if kept flat in baseline, document as deliberate choice and a deviation from paper's wording.

#### B7. Key image selection (Section 3.2.1) ✅ fixed
- **Where:** `scripts/01_select_orb_key_images.py`; `run_town01_paper_pipeline.sh` Step 2-4
- **Paper 3.2.1:** ORB-feature-based key image selection from the first-pass image set before COLMAP SfM, to reduce redundancy and speed up reconstruction.
- **Now:** first-pass images are filtered through ORB key-image selection before COLMAP coarse reconstruction. If `poses.json` exists, it is filtered to the same key-image subset.
- **Controls:** enabled by default; set `USE_ORB_KEY_IMAGES=0` to bypass. Thresholds are controlled by `ORB_MATCH_THRESHOLD`, `ORB_MIN_FEATURES`, `ORB_MAX_FEATURES`, and `ORB_MAX_SIDE`.
- **Verification:** synthetic image/poses check passed 2026-05-06; selected key images and filtered pose frames matched.

---

## 🌱 Course-improvement candidates

Branch off `main` after the baseline is paper-faithful. See `course_improvement_ideas.md` for full write-ups.

| # | Idea | Status |
|---|---|---|
| C1 | Quality-aware local view count (S_l samples get 2 local views instead of 1) | Proposed (was previous code's behaviour) |
| C2 | Replace Poisson-recon-based quality scoring with learned/geometric alternative | Brainstormed, not detailed |
| C3 | Submodular / learning-based viewpoint selection (vs greedy) | Brainstormed, not detailed |
| C4 | Replace SfM+MVS with 3DGS for the second-pass reconstruction | Brainstormed (gennbv_blackwall infra reusable) |
| C5 | Co-optimize TSP with reconstruction-quality predictor | Brainstormed, not detailed |

---

## Recommended next-step order

(Skipping a Town01 smoke test — pipeline is too long; verify each fix in isolation instead.)

1. **Optional strictness work:** B6 (normal/location clusters) — defer unless strict-reproducibility pressure demands.
2. **When ready for numbers:** run a reduced smoke test first, then the full Town01/Town03 pipeline.
3. **Branch for course project,** start with C1 to establish the comparison harness, then pick a more substantive one (C2 / C3 / C4).

---

## Assumptions and unknown parameters

Paper does not specify (or only hand-waves) several values. Document the choice for each here so the writeup can list them as assumptions rather than implicit decisions.

| Parameter | Paper's specification | Current value / status |
|---|---|---|
| `θ_t` (orientation angle threshold, used in Eq 2 and Eq 7) | not given | 60° = π/3; assumption recorded in `Params*.h` |
| Quality threshold (S_l membership cutoff) | not given; paper sorts ascending and "selects an ascending set whose score is lower than a pre-defined threshold". Paper Table 1 reports #Local ≈ 110-120 viewpoints per scene → S_l is a small minority. | 10th percentile (`main.cpp:35` `samplepoints[len / 10].quality`) — chosen to match paper's reported viewpoint counts. Produces ~210 total viewpoints on town01 (paper Building-1: 210; Building-2: 205). Was 80th percentile pre-2026-05-22 (produced 1581 viewpoints, 5-15× paper). |
| Poisson-disk sampling radius `R` / quality neighbourhood radius | not given | sampling radius is derived from mesh area and `--num-samples`; quality `R` is `--quality-radius` (default 3.0 m) |
| `VOXEL_SIZE` (view sampling space voxel) | not given | 10.0 (m) |
| `dmin`, `dmax` (view sampling distance shell) | not given numerically; paper just says "specified safe distances" | 25 / 35 (m) |
| `BEST_DISTANCE` (`d_opt` in Eq 6) | paper says `(d_min + d_max) / 2` | 30 (m) — matches `(25+35)/2` ✓ |
| Camera FOV (used in B4 if FOV-radius mark adopted) | not given experimentally | 90° H × 56.25° V (`Params.h`) — must match the CARLA camera used to render the second-pass images |
| Global scanned mark mode | paper uses FOV-radius mark | `USE_PAPER_FOV_MARK=true` by default; set false for ray-visibility variant |
| ACO hyperparameters (α, β, ρ, ant count, iterations) | references [34, 35]; no values given | α=1.0, β=2.0, ρ=0.1, ants=20, iterations=100; override via `ACO_*` env vars |
| "Adjacent" definition for global exclusion | not given | 6-connected with `VOXEL_SIZE × 1.1` slack — document |
| TSP tour closure | paper calls it TSP but does not state return-to-start | open Hamiltonian ordering; assumption for UAV mission |
| Cubic Bezier smoothing density | paper says cubic Bezier, no sampling density | `ACO_SMOOTH_SAMPLES` default 4 samples/segment; `ACO_SMOOTH_TENSION` default 0.35 |
| ORB key-image selection thresholds | paper describes thresholds but does not publish values | `ORB_MATCH_THRESHOLD=450`, `ORB_MIN_FEATURES=150`, `ORB_MAX_FEATURES=4000`, `ORB_MAX_SIDE=1600`; override via env vars |

---

## Open questions

*(none currently — the per-sample orientation question was resolved 2026-05-06: paper dedupes by voxel; orientation is bound to the trigger sample s_j of each selection.)*
