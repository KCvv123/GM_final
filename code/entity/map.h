//
// Created by lj on 2022/11/10.
//

#ifndef FUXIAN_MAP_H
#define FUXIAN_MAP_H
#include "iostream"
#include "vector"
#include <Eigen/Dense>
#include "sample_point.h"

class Map {
private:
    float xMin;
    float xMax;
    float yMin;
    float yMax;
    float zMin;
    float zMax;
    float xOffset;       // x偏移量
    float yOffset;       // y偏移量
    float xResolution;   // x分辨率
    float yResolution;   // y分辨率
    int mapWidth;
    int mapHeight;
    std::vector<std::vector<float>> heightMap;

    // OBB (Oriented Bounding Box) — Paper Section 3.2.2
    // PCA on normals → 3 axes; one axis constrained to vertical
    Eigen::Vector3f obbAxes[3];    // 3 orthonormal axes (axes[2] = vertical)
    Eigen::Vector3f obbCenter;     // center of OBB in world coords
    Eigen::Vector3f obbHalfExt;    // half-extents along each axis

    void initBoundary(std::vector<SamplePoint>& points);
    void initHeightMap(std::vector<SamplePoint>& points);
    void initOBB(std::vector<SamplePoint>& points);

public:
    Map(std::vector<SamplePoint>& points, float xResolution=10.0, float yResolution=10.0);
    float getXOffset();
    float getYOffset();
    int getXResolution();
    int getYResolution();
    int getMapWidth();
    int getMapHeight();
    float getMapData(int row, int col);
    float getMinHeightOfPos(float xPos, float yPos);

    // AABB (kept for height map queries)
    float getXMin() const { return xMin; }
    float getXMax() const { return xMax; }
    float getYMin() const { return yMin; }
    float getYMax() const { return yMax; }
    float getZMin() const { return zMin; }
    float getZMax() const { return zMax; }
    bool isInsideBBox(float x, float y, float z) const {
        return x >= xMin && x <= xMax &&
               y >= yMin && y <= yMax &&
               z >= zMin;
    }

    // OBB accessors
    const Eigen::Vector3f* getOBBAxes() const { return obbAxes; }
    const Eigen::Vector3f& getOBBCenter() const { return obbCenter; }
    const Eigen::Vector3f& getOBBHalfExtents() const { return obbHalfExt; }

    // Project world point into OBB local coords
    Eigen::Vector3f worldToOBB(float x, float y, float z) const {
        Eigen::Vector3f p(x - obbCenter.x(), y - obbCenter.y(), z - obbCenter.z());
        return Eigen::Vector3f(p.dot(obbAxes[0]), p.dot(obbAxes[1]), p.dot(obbAxes[2]));
    }

    // OBB local coords back to world
    Eigen::Vector3f obbToWorld(float a0, float a1, float a2) const {
        return obbCenter + a0 * obbAxes[0] + a1 * obbAxes[1] + a2 * obbAxes[2];
    }
};


#endif //FUXIAN_MAP_H
