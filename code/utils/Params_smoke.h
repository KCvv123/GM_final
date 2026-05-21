//
// Tiny synthetic smoke-test parameters for Step 5-7 wiring only.
//
#ifndef FUXIAN_PARAMS_H
#define FUXIAN_PARAMS_H
#define M_PI 3.1415926

namespace Params{
    const std::string SAMPLE_FILE_PATH = "data/smoke_samples.ply";
    const std::string MESH_FILE_PATH   = "data/smoke_mesh.ply";

    const float MIN_DISTANCE_BETWEEN_POINT_AND_VIEW = 5;
    const float MAX_DISTANCE_BETWEEN_POINT_AND_VIEW = 12;
    const float PER_RADIAN = 360 / 36.0 / 180.0 * M_PI;

    const int POP_SIZE = 10;
    const int TOTAL_VIEW_NUMS = 120;

    const float VOXEL_SIZE = 5.0f;
    const float MAX_D = 50;
    const float BEST_DISTANCE = 8.5f;

    const float THETA_T = M_PI / 3.0f;
    const bool USE_PAPER_FOV_MARK = true;

    const double RESOLUTION_H = 1280;
    const double RESOLUTION_V = 720;
    const double FOV_H        = 90;
    const double FOV_V        = 56.25;
}

#endif //FUXIAN_PARAMS_H
