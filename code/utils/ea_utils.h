//
// Created by lj on 2022/11/9.
//

#ifndef FUXIAN_EA_UTILS_H
#define FUXIAN_EA_UTILS_H

#include <Eigen/Dense>
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
    static void selectViewPointByScore(ScoreUtils &scoreUtils,
                                       std::vector<ViewSelection> &finalSelections,
                                       std::vector<SamplePoint> &points,
                                       const std::vector<Eigen::Vector3f> &candidatePositions,
                                       float qualityThreshold);

    // Build candidate pool + greedy solution + precomputed visibility
    static EAContext initPopulationWithContext(ScoreUtils &scoreUtils, Map &map,
                                              std::vector<SamplePoint> &points,
                                              float qualityThreshold);

    // Convert selections to ViewPoints; orientation = sample - viewpoint position,
    // where sample is the trigger sample bound at selection time.
    static std::vector<ViewPoint> selectionsToViewPoints(
        const std::vector<ViewSelection> &selections,
        const std::vector<Eigen::Vector3f> &candidatePositions,
        std::vector<SamplePoint> &points);
};


#endif //FUXIAN_EA_UTILS_H
