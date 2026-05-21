//
// Created by lj on 2022/11/9.
// Rewritten for paper-faithful voxel-grid viewpoint candidates (Section 3.3.1)
//

#include "ea_utils.h"
#include "compare.h"
#include "ply_utils.h"
#include "log.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// =============================================================================
// Paper Section 3.3.1: Voxel-grid candidate generation
// Voxelize the OBB expanded by dmax; keep voxels whose nearest sample
// distance is in [dmin, dmax].
// =============================================================================

std::vector<Eigen::Vector3f> EAUtils::generateVoxelCandidates(
        Map &map, std::vector<SamplePoint> &points) {

    const float voxelSize = Params::VOXEL_SIZE;
    const float dmin = Params::MIN_DISTANCE_BETWEEN_POINT_AND_VIEW;
    const float dmax = Params::MAX_DISTANCE_BETWEEN_POINT_AND_VIEW;
    const float dmin2 = dmin * dmin;
    const float dmax2 = dmax * dmax;

    // Expand OBB by dmax in all directions
    Eigen::Vector3f halfExt = map.getOBBHalfExtents();
    float ext0 = halfExt[0] + dmax;
    float ext1 = halfExt[1] + dmax;
    float ext2 = halfExt[2] + dmax;

    // Number of voxels along each OBB axis
    int n0 = (int)std::ceil(2.0f * ext0 / voxelSize);
    int n1 = (int)std::ceil(2.0f * ext1 / voxelSize);
    int n2 = (int)std::ceil(2.0f * ext2 / voxelSize);

    std::cout << "[voxel] Grid: " << n0 << " x " << n1 << " x " << n2
              << " = " << n0*n1*n2 << " voxels (before filtering)" << std::endl;

    // Build sample position array for fast distance queries
    int nSamples = (int)points.size();
    std::vector<Eigen::Vector3f> samplePos(nSamples);
    for (int i = 0; i < nSamples; i++) {
        float *p = points[i].getPos();
        samplePos[i] = Eigen::Vector3f(p[0], p[1], p[2]);
    }

    std::vector<Eigen::Vector3f> candidates;
    int totalChecked = 0;

    for (int i0 = 0; i0 < n0; i0++) {
        float a0 = -ext0 + (i0 + 0.5f) * voxelSize;
        for (int i1 = 0; i1 < n1; i1++) {
            float a1 = -ext1 + (i1 + 0.5f) * voxelSize;
            for (int i2 = 0; i2 < n2; i2++) {
                float a2 = -ext2 + (i2 + 0.5f) * voxelSize;
                totalChecked++;

                // Convert OBB-local coords to world
                Eigen::Vector3f worldPos = map.obbToWorld(a0, a1, a2);

                // Reject voxels below local terrain height
                float terrainZ = map.getMinHeightOfPos(worldPos.x(), worldPos.y());
                if (terrainZ <= -1.0f) terrainZ = 0.0f;  // out of bounds or no data
                if (worldPos.z() < terrainZ) continue;

                // Find squared distance to nearest sample
                float minDist2 = std::numeric_limits<float>::max();
                for (int s = 0; s < nSamples; s++) {
                    float d2 = (worldPos - samplePos[s]).squaredNorm();
                    if (d2 < minDist2) minDist2 = d2;
                    if (d2 < dmin2) break;  // too close, skip early
                }

                // Keep if dmin <= dist <= dmax
                if (minDist2 >= dmin2 && minDist2 <= dmax2) {
                    candidates.push_back(worldPos);
                }
            }
        }
    }

    std::cout << "[voxel] Candidates after distance filter: " << candidates.size()
              << " / " << totalChecked << std::endl;

    return candidates;
}

// =============================================================================
// Paper Section 3.3.2: Quality-aware greedy viewpoint selection
// Visibility must be precomputed via scoreUtils.updateVisibilityRayOnly()
// before calling this function.
// =============================================================================

// Helper: check if a viewpoint index is already in the selections list.
static bool viewIdxAlreadySelected(int vIdx,
                                   const std::vector<ViewSelection> &selections) {
    for (const auto &sel : selections) {
        if (sel.viewIdx == vIdx) return true;
    }
    return false;
}

// Helper: voxel-adjacency test for global-stage exclusion (Paper §3.3.2).
// 6-connected face-neighbours, with floating-point slack.
static bool isAdjacentToAnySelected(int vIdx,
                                    const std::vector<ViewSelection> &selections,
                                    const std::vector<Eigen::Vector3f> &candidatePositions,
                                    float adjDist2) {
    if (vIdx < 0 || vIdx >= (int)candidatePositions.size()) return false;
    const Eigen::Vector3f &p = candidatePositions[vIdx];
    for (const auto &sel : selections) {
        if (sel.viewIdx == vIdx) continue;
        if (sel.viewIdx < 0 || sel.viewIdx >= (int)candidatePositions.size()) continue;
        float d2 = (p - candidatePositions[sel.viewIdx]).squaredNorm();
        if (d2 < adjDist2) return true;
    }
    return false;
}

// Paper Section 3.3.2: mark samples near the trigger sample as scanned using
// |d(v_l, s_j)| * sin(theta_fov / 2). This is the paper-literal alternative to
// the Embree ray-visibility coverage map.
static void markPaperFovCoverage(const ViewSelection &sel,
                                 std::vector<int> &coverage,
                                 std::vector<SamplePoint> &points,
                                 const std::vector<Eigen::Vector3f> &candidatePositions) {
    if (sel.viewIdx < 0 || sel.viewIdx >= (int)candidatePositions.size()) return;
    if (sel.triggerSample < 0 || sel.triggerSample >= (int)points.size()) return;

    const Eigen::Vector3f &vPos = candidatePositions[sel.viewIdx];
    float *triggerPosRaw = points[sel.triggerSample].getPos();
    Eigen::Vector3f triggerPos(triggerPosRaw[0], triggerPosRaw[1], triggerPosRaw[2]);
    float radius = (vPos - triggerPos).norm()
                   * std::sin((float)Params::FOV_H * (float)M_PI / 180.0f / 2.0f);
    float radius2 = radius * radius;

    for (int i = 0; i < (int)points.size(); ++i) {
        float *pRaw = points[i].getPos();
        Eigen::Vector3f p(pRaw[0], pRaw[1], pRaw[2]);
        if ((p - triggerPos).squaredNorm() <= radius2) {
            coverage[i]++;
        }
    }
}

void EAUtils::selectViewPointByScore(ScoreUtils &scoreUtils,
                                     std::vector<ViewSelection> &finalSelections,
                                     std::vector<SamplePoint> &points,
                                     const std::vector<Eigen::Vector3f> &candidatePositions,
                                     float qualityThreshold) {
    int len = (int)points.size();
    float dom_2 = pow(Params::BEST_DISTANCE - Params::MIN_DISTANCE_BETWEEN_POINT_AND_VIEW, 2);
    const float cosThetaT = std::cos(Params::THETA_T);
    const float thetaDelta = 1.0f - cosThetaT;
    const float thetaDenom = std::max(thetaDelta * thetaDelta, 1e-6f);

    // === LOCAL VIEWPOINT SELECTION (Paper §3.3.2) ===
    // S_l = samples with quality < threshold; pick 1 best unchosen viewpoint per sample.
    // High-quality samples get 0 local views — covered only by global stage.
    int s_l_count = 0;
    for (int i = 0; i < len; ++i) {
        if (points[i].quality >= qualityThreshold) continue;  // S_l only
        s_l_count++;

        std::vector<ViewScore> score1;
        for (int j = 0; j < (int)scoreUtils.pointViewVisibilitySet[i].size(); ++j) {
            int vIdx = scoreUtils.pointViewVisibilitySet[i][j];
            if (vIdx < 0 || vIdx >= (int)candidatePositions.size()) continue;

            // Distance confidence c_d (Paper Eq 6)
            float *pPos = points[i].getPos();
            float dx = candidatePositions[vIdx].x() - pPos[0];
            float dy = candidatePositions[vIdx].y() - pPos[1];
            float dz = candidatePositions[vIdx].z() - pPos[2];
            float dis_ij = std::sqrt(dx*dx + dy*dy + dz*dz);
            float cd = 1.0f - pow(dis_ij - Params::BEST_DISTANCE, 2) / dom_2;
            if (cd < 0.0f) cd = 0.0f;

            // Orientation confidence c_o (Paper Eq 7)
            float *ns = points[i].getDirection();
            float dij[3] = {dx, dy, dz};
            float dij_len = dis_ij;
            if (dij_len < 1e-6f) continue;
            dij[0] /= dij_len; dij[1] /= dij_len; dij[2] /= dij_len;
            float nd = ns[0]*dij[0] + ns[1]*dij[1] + ns[2]*dij[2];
            if (nd < cosThetaT) continue;
            float co = exp(-1.0f * pow((1.0f - nd), 2) / thetaDenom);

            ViewScore s(vIdx, i, cd * co);
            score1.emplace_back(std::move(s));
        }

        sort(score1.begin(), score1.end(), Compare::compareByViewScoreFromBigToSmall);

        for (int k = 0; k < (int)score1.size(); ++k) {
            if (!viewIdxAlreadySelected(score1[k].viewIndex, finalSelections)) {
                finalSelections.push_back({score1[k].viewIndex, i});
                break;  // 1 viewpoint per S_l sample
            }
        }
    }
    std::cout << "[select] S_l: " << s_l_count << " samples (1 view each)" << std::endl;
    std::cout << "[select] Local viewpoints: " << finalSelections.size() << std::endl;

    // === GLOBAL VIEWPOINT SELECTION (Paper §3.3.2) ===
    // All samples need scanned >= 2; exclude already-selected AND adjacent candidates
    // (6-connected voxel neighbours) for spatial uniformity.
    const int globalMinCov = 2;
    const float adjThresh  = Params::VOXEL_SIZE * 1.1f;
    const float adjDist2   = adjThresh * adjThresh;

    std::vector<int> coverage(len, 0);
    for (const auto &sel : finalSelections) {
        if (Params::USE_PAPER_FOV_MARK) {
            markPaperFovCoverage(sel, coverage, points, candidatePositions);
        } else {
            if (sel.viewIdx < 0 || sel.viewIdx >= (int)scoreUtils.viewPointVisibilitySet.size()) continue;
            for (int pIdx : scoreUtils.viewPointVisibilitySet[sel.viewIdx]) {
                if (pIdx >= 0 && pIdx < len) coverage[pIdx]++;
            }
        }
    }
    std::cout << "[select] Coverage mode: "
              << (Params::USE_PAPER_FOV_MARK ? "paper FOV-radius mark" : "ray visibility")
              << std::endl;

    int globalAdded = 0;
    for (int i = 0; i < len; ++i) {
        if (coverage[i] >= globalMinCov) continue;
        int needed = globalMinCov - coverage[i];

        std::vector<ViewScore> score1;
        for (int j = 0; j < (int)scoreUtils.pointViewVisibilitySet[i].size(); ++j) {
            int vIdx = scoreUtils.pointViewVisibilitySet[i][j];
            if (vIdx < 0 || vIdx >= (int)candidatePositions.size()) continue;

            float *pPos = points[i].getPos();
            float dx = candidatePositions[vIdx].x() - pPos[0];
            float dy = candidatePositions[vIdx].y() - pPos[1];
            float dz = candidatePositions[vIdx].z() - pPos[2];
            float dis_ij = std::sqrt(dx*dx + dy*dy + dz*dz);
            float cdval = 1.0f - pow(dis_ij - Params::BEST_DISTANCE, 2) / dom_2;
            if (cdval < 0.0f) cdval = 0.0f;

            float *ns = points[i].getDirection();
            float dij[3] = {dx, dy, dz};
            float dij_len = dis_ij;
            if (dij_len < 1e-6f) continue;
            dij[0] /= dij_len; dij[1] /= dij_len; dij[2] /= dij_len;
            float nd = ns[0]*dij[0] + ns[1]*dij[1] + ns[2]*dij[2];
            if (nd < cosThetaT) continue;
            float co = exp(-1.0f * pow((1.0f - nd), 2) / thetaDenom);

            ViewScore s(vIdx, i, cdval * co);
            score1.emplace_back(std::move(s));
        }
        sort(score1.begin(), score1.end(), Compare::compareByViewScoreFromBigToSmall);

        int added = 0;
        for (int k = 0; k < (int)score1.size() && added < needed; ++k) {
            int vIdx = score1[k].viewIndex;
            if (viewIdxAlreadySelected(vIdx, finalSelections)) continue;
            if (isAdjacentToAnySelected(vIdx, finalSelections, candidatePositions, adjDist2)) continue;
            ViewSelection selected{vIdx, i};
            finalSelections.push_back(selected);
            if (Params::USE_PAPER_FOV_MARK) {
                markPaperFovCoverage(selected, coverage, points, candidatePositions);
            } else {
                if (vIdx >= 0 && vIdx < (int)scoreUtils.viewPointVisibilitySet.size()) {
                    for (int pIdx : scoreUtils.viewPointVisibilitySet[vIdx]) {
                        if (pIdx >= 0 && pIdx < len) coverage[pIdx]++;
                    }
                }
            }
            added++;
            globalAdded++;
        }
    }
    std::cout << "[select] Global viewpoints added: " << globalAdded << std::endl;
    std::cout << "[select] Total viewpoints: " << finalSelections.size() << std::endl;
}

// =============================================================================
// Convert selections to ViewPoints. Orientation is bound to the trigger sample
// recorded at selection time (Paper Section 3.3.2).
// =============================================================================

std::vector<ViewPoint> EAUtils::selectionsToViewPoints(
        const std::vector<ViewSelection> &selections,
        const std::vector<Eigen::Vector3f> &candidatePositions,
        std::vector<SamplePoint> &points) {

    std::vector<ViewPoint> result;
    result.reserve(selections.size());

    for (const auto &sel : selections) {
        if (sel.viewIdx < 0 || sel.viewIdx >= (int)candidatePositions.size()) continue;
        const Eigen::Vector3f &vPos = candidatePositions[sel.viewIdx];

        if (sel.triggerSample >= 0 && sel.triggerSample < (int)points.size()) {
            float *sPos = points[sel.triggerSample].getPos();
            // INWARD direction: trigger sample - viewpoint (camera looks at the sample
            // that triggered this selection — Paper 3.3.2 binds orientation to s_j)
            float dirX = sPos[0] - vPos.x();
            float dirY = sPos[1] - vPos.y();
            float dirZ = sPos[2] - vPos.z();
            result.emplace_back(vPos.x(), vPos.y(), vPos.z(), dirX, dirY, dirZ);
        } else {
            // Should not happen if selectViewPointByScore always sets a trigger sample.
            result.emplace_back(vPos.x(), vPos.y(), vPos.z(), 0, 0, -1);
        }
    }
    return result;
}

// =============================================================================
// initPopulationWithContext — voxel grid candidates + ray-only visibility
// =============================================================================

EAContext EAUtils::initPopulationWithContext(ScoreUtils &scoreUtils, Map &map,
                                            std::vector<SamplePoint> &points,
                                            float qualityThreshold) {
    EAContext ctx;

    // Step 1: Generate voxel grid candidates (Paper 3.3.1)
    ctx.candidatePositions = generateVoxelCandidates(map, points);

    // Step 2: Compute ray-only visibility (no frustum check — Paper w_v)
    std::cout << "[init] Computing ray-only visibility for "
              << ctx.candidatePositions.size() << " candidates x "
              << points.size() << " samples..." << std::endl;
    scoreUtils.updateVisibilityRayOnly(ctx.candidatePositions);

    // Step 3: Greedy selection with quality-aware S_high/S_low split
    selectViewPointByScore(scoreUtils, ctx.greedySelections,
                           points, ctx.candidatePositions, qualityThreshold);

    std::cout << "[init] Greedy solution: " << ctx.greedySelections.size() << " viewpoints" << std::endl;

    return ctx;
}
