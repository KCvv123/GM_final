#include <iostream>
#include <Eigen/Geometry>
#include <vector>
#include "entity/sample_point.h"
#include "utils/io_utils.h"
#include "utils/Params.h"
#include "math.h"
#include "utils/tools.h"
#include "utils/score.h"
#include "utils/log.h"
#include "entity/map.h"
#include "utils/ea_utils.h"
#include <CGAL/Point_set_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <algorithm>
#include "utils/compare.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> SurfaceMesh;
typedef CGAL::Point_set_3<Point_3> PointSet;
typedef SurfaceMesh::Vertex_range VertexRange;
typedef SurfaceMesh::Vertex_index VertexIndex;
typedef SurfaceMesh::Face_index FaceIndex;


int main() {
    std::cout << "FuXian Path Planning (117-paper)" << std::endl;

    // Read sample points WITH quality (type=1: sorted ascending by quality internally)
    std::vector<SamplePoint> samplepoints;
    IOUtils::readPlyFile(Params::SAMPLE_FILE_PATH, samplepoints, 1);
    int len = samplepoints.size();
    // Paper §3.3.2: S_l = "ascending set whose score is lower than a pre-defined threshold".
    // Paper Table 1 reports #Local ~110-120 viewpoints per scene → S_l is a small minority.
    // 10th percentile chosen to match paper's Building-1/2/3, Real-1, City-1 viewpoint counts.
    float threshold = samplepoints[len / 10].quality;
    std::cout << "[main] " << len << " sample points, quality threshold (10th pct) = " << threshold << std::endl;

    // Read mesh for Embree ray tracing
    std::vector<SamplePoint> meshPoints;
    std::vector<int*> faceIndexes;
    IOUtils::readPlyForAll(Params::MESH_FILE_PATH, meshPoints, faceIndexes);

    ScoreUtils scoreUtils(faceIndexes, meshPoints, samplepoints);
    Map map(samplepoints);
    Log::setLog(true, true, false);

    std::cout << "[main] Generating candidates + greedy selection..." << std::endl;

    // Method dispatch via env vars (course-improvement: confidence_coverage adds
    // a budgeted submodular planner; default keeps the paper-faithful Yan two-stage).
    const char* method_env = std::getenv("FUXIAN_METHOD");
    std::string method = method_env ? method_env : "yan_two_stage";
    const char* k_env = std::getenv("FUXIAN_K");
    int budgetK = k_env ? std::atoi(k_env) : -1;
    const char* hreq_env = std::getenv("FUXIAN_H_REQ");
    float hReq = hreq_env ? (float)std::atof(hreq_env) : 2.0f;

    // Generate candidate viewpoints + greedy selection (Paper 3.3 or proposal §3.2)
    EAContext ctx = EAUtils::initPopulationWithContext(scoreUtils, map, samplepoints,
                                                      threshold, method, budgetK, hReq);

    // Convert greedy selections to ViewPoints; orientation bound to trigger sample (Paper 3.3.2)
    std::vector<ViewPoint> bestViewpoints = EAUtils::selectionsToViewPoints(
        ctx.greedySelections, ctx.candidatePositions, samplepoints);

    // Output. Suffix by method so the three planners' outputs co-exist:
    //   yan_two_stage      -> <name>.txt           (paper-faithful baseline)
    //   binary_coverage    -> <name>_bc.txt        (proposal §3.1 controlled baseline)
    //   confidence_coverage-> <name>_cwc.txt       (proposal §3.2 method)
    const char* tag_env = std::getenv("FUXIAN_TAG");
    std::string tag  = tag_env ? tag_env : "town01";
    const char* name_env = std::getenv("FUXIAN_NAME");
    std::string name = name_env ? name_env : "town01_viewpoints";
    if (method == "confidence_coverage")      name += "_cwc";
    else if (method == "binary_coverage")     name += "_bc";

    IOUtils::saveViewPoint("output", tag, name, bestViewpoints);
    std::string savepath = "output";
    IOUtils::viewPointsToSmithPath(bestViewpoints, savepath);

    std::cout << "[main] Done. " << bestViewpoints.size() << " viewpoints saved." << std::endl;
    return 0;
}
