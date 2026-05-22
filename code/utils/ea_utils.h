//
// Created by lj on 2022/11/9.
//

#ifndef FUXIAN_EA_UTILS_H
#define FUXIAN_EA_UTILS_H

#include <Eigen/Dense>
#include <string>
#include "score.h"
#include "Params.h"
#include "../entity/view_point.h"
#include "../entity/sample_point.h"
#include "../entity/map.h"

// A selected viewpoint: position index + the sample that triggered its selection.
// Paper 3.3.2: orientation o_i is bound to the sample that triggered the choice.
struct ViewSelection {
    int viewIdx;        // index into EAContext::candidatePositions
    int triggerSample;  // index into points; defines viewpoint orientation (Paper 3.3.2)
};

struct EAContext {
    std::vector<Eigen::Vector3f> candidatePositions;
    std::vector<ViewSelection> greedySelections;
};

class EAUtils {
public:
    // Paper Section 3.3.1: voxel grid candidate generation
    static std::vector<Eigen::Vector3f> generateVoxelCandidates(
        Map &map, std::vector<SamplePoint> &points);

    // Quality-aware greedy selection (Paper Section 3.3.2)
    // Each selection records both the viewpoint index AND the sample that triggered it,
    // so orientation can be bound to that trigger sample (Paper 3.3.2).
    // budgetK > 0 caps the total number of selections (for budget-matched comparison
    // with the course-improvement CWC method). budgetK < 0 = no cap (paper-faithful).
    static void selectViewPointByScore(ScoreUtils &scoreUtils,
                                       std::vector<ViewSelection> &finalSelections,
                                       std::vector<SamplePoint> &points,
                                       const std::vector<Eigen::Vector3f> &candidatePositions,
                                       float qualityThreshold,
                                       int budgetK = -1);

    // Confidence-weighted coverage greedy (Course improvement, Proposal Section 3.2).
    // For each greedy step, picks the candidate viewpoint that maximises
    //   Σ_s min( max(0, H_req - H(s)), c(v, s) )
    // where H(s) accumulates effective per-sample coverage Σ_{v∈V_sel} c(v, s).
    // This is a monotone submodular objective; greedy gives a (1-1/e) approximation
    // guarantee (Nemhauser 1978). Replaces the paper's binary scanned-count rule.
    static void selectViewPointByConfidenceCoverage(
        ScoreUtils &scoreUtils,
        std::vector<ViewSelection> &finalSelections,
        std::vector<SamplePoint> &points,
        const std::vector<Eigen::Vector3f> &candidatePositions,
        int budgetK,
        float hReq);

    // Build candidate pool + greedy solution + precomputed visibility.
    // If method == "confidence_coverage", uses CWC; otherwise paper Yan two-stage.
    static EAContext initPopulationWithContext(ScoreUtils &scoreUtils, Map &map,
                                              std::vector<SamplePoint> &points,
                                              float qualityThreshold,
                                              const std::string &method = "yan_two_stage",
                                              int budgetK = -1,
                                              float hReq = 2.0f);

    // Convert selections to ViewPoints; orientation = sample - viewpoint position,
    // where sample is the trigger sample bound at selection time.
    static std::vector<ViewPoint> selectionsToViewPoints(
        const std::vector<ViewSelection> &selections,
        const std::vector<Eigen::Vector3f> &candidatePositions,
        std::vector<SamplePoint> &points);
};


#endif //FUXIAN_EA_UTILS_H
