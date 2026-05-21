//
// Params for CARLA Town03 experiment.
// Copy to Params.h before building:
//   cp code/utils/Params_town03.h code/utils/Params.h
//
#ifndef FUXIAN_PARAMS_H
#define FUXIAN_PARAMS_H
#define M_PI 3.1415926

namespace Params{
    // --- Data paths ---
    const std::string SAMPLE_FILE_PATH = "data/town03_samples.ply";
    const std::string MESH_FILE_PATH   = "data/town03_mesh.ply";

    // --- Viewpoint candidate distances ---
    const float MIN_DISTANCE_BETWEEN_POINT_AND_VIEW = 25;
    const float MAX_DISTANCE_BETWEEN_POINT_AND_VIEW = 35;

    // Angular step for horizontal sampling (10° per step, 36 steps = 360°)
    const float PER_RADIAN = 360 / 36.0 / 180.0 * M_PI;

    // --- EA population size ---
    const int POP_SIZE = 50;

    // Maximum number of viewpoints that can appear in a single individual.
    const int TOTAL_VIEW_NUMS = 2000;

    // Voxel size for viewpoint candidate grid (Paper Section 3.3.1)
    const float VOXEL_SIZE = 10.0f;

    // Maximum camera observation distance (metres).
    const float MAX_D = 200;

    // Ideal distance from sample point to camera
    const float BEST_DISTANCE = 30;

    // Paper Eq 2 / Eq 7 normal-view angle threshold. The paper does not
    // publish a value; 60 degrees is recorded as the baseline assumption.
    const float THETA_T = M_PI / 3.0f;

    // Paper-literal global scanned mark (Section 3.3.2) uses an FOV-radius
    // neighbourhood around the trigger sample. Set false for ray-visibility variant.
    const bool USE_PAPER_FOV_MARK = true;

    // --- Camera intrinsics ---
    const double RESOLUTION_H = 1280;
    const double RESOLUTION_V = 720;
    const double FOV_H        = 90;
    const double FOV_V        = 56.25;
}

#endif //FUXIAN_PARAMS_H
