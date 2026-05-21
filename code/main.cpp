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
    float threshold = samplepoints[len * 4 / 5].quality;
    std::cout << "[main] " << len << " sample points, quality threshold (80th pct) = " << threshold << std::endl;

    // Read mesh for Embree ray tracing
    std::vector<SamplePoint> meshPoints;
    std::vector<int*> faceIndexes;
    IOUtils::readPlyForAll(Params::MESH_FILE_PATH, meshPoints, faceIndexes);

    ScoreUtils scoreUtils(faceIndexes, meshPoints, samplepoints);
    Map map(samplepoints);
    Log::setLog(true, true, false);

    std::cout << "[main] Generating candidates + greedy selection..." << std::endl;

    // Generate candidate viewpoints + quality-aware greedy selection (Paper 3.3)
    EAContext ctx = EAUtils::initPopulationWithContext(scoreUtils, map, samplepoints, threshold);

    // Convert greedy selections to ViewPoints; orientation bound to trigger sample (Paper 3.3.2)
    std::vector<ViewPoint> bestViewpoints = EAUtils::selectionsToViewPoints(
        ctx.greedySelections, ctx.candidatePositions, samplepoints);

    // Output
    const char* tag_env = std::getenv("FUXIAN_TAG");
    std::string tag  = tag_env ? tag_env : "town01";
    const char* name_env = std::getenv("FUXIAN_NAME");
    std::string name = name_env ? name_env : "town01_viewpoints";

    IOUtils::saveViewPoint("output", tag, name, bestViewpoints);
    std::string savepath = "output";
    IOUtils::viewPointsToSmithPath(bestViewpoints, savepath);

    std::cout << "[main] Done. " << bestViewpoints.size() << " viewpoints saved." << std::endl;
    return 0;
}
