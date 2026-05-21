//
// Created by lj on 2022/11/10.
//

#include "map.h"
#include <cfloat>
#include <cmath>
#include <iostream>

Map::Map(std::vector<SamplePoint> &points, float xResolution, float yResolution) {
    this->xResolution = xResolution;
    this->yResolution = yResolution;
    initBoundary(points);
    initHeightMap(points);
    initOBB(points);
}

void Map::initOBB(std::vector<SamplePoint> &points) {
    int n = (int)points.size();
    if (n == 0) return;

    // Paper 3.2.2: PCA on normals to find dominant axes
    Eigen::MatrixXf normals(n, 3);
    Eigen::MatrixXf positions(n, 3);
    for (int i = 0; i < n; i++) {
        float *d = points[i].getDirection();
        float *p = points[i].getPos();
        normals(i, 0) = d[0]; normals(i, 1) = d[1]; normals(i, 2) = d[2];
        positions(i, 0) = p[0]; positions(i, 1) = p[1]; positions(i, 2) = p[2];
    }

    // Center normals and compute covariance
    Eigen::Vector3f meanN = normals.colwise().mean();
    Eigen::MatrixXf centered = normals.rowwise() - meanN.transpose();
    Eigen::Matrix3f cov = (centered.transpose() * centered) / (float)(n - 1);

    // Eigen decomposition — columns sorted by eigenvalue ascending
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    Eigen::Matrix3f eigVecs = solver.eigenvectors();

    // axes[2] = vertical (constrained to gravity direction)
    obbAxes[2] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);

    // Find which eigenvector is most aligned with Z and replace it
    int zIdx = 0;
    float maxDot = 0;
    for (int i = 0; i < 3; i++) {
        float d = std::abs(eigVecs.col(i).dot(obbAxes[2]));
        if (d > maxDot) { maxDot = d; zIdx = i; }
    }

    // The other two eigenvectors become horizontal axes
    int hIdx = 0;
    for (int i = 0; i < 3; i++) {
        if (i == zIdx) continue;
        Eigen::Vector3f axis = eigVecs.col(i);
        axis.z() = 0.0f;
        float len = axis.norm();
        if (len > 1e-6f) axis /= len;
        else axis = (hIdx == 0) ? Eigen::Vector3f(1, 0, 0) : Eigen::Vector3f(0, 1, 0);
        obbAxes[hIdx] = axis;
        hIdx++;
    }

    // Re-orthogonalize: axes[1] = axes[2] × axes[0]
    obbAxes[1] = obbAxes[2].cross(obbAxes[0]).normalized();
    obbAxes[0] = obbAxes[1].cross(obbAxes[2]).normalized();

    // Project all positions onto OBB axes to get extents
    float minExt[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float maxExt[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (int i = 0; i < n; i++) {
        Eigen::Vector3f pos(positions(i, 0), positions(i, 1), positions(i, 2));
        for (int a = 0; a < 3; a++) {
            float proj = pos.dot(obbAxes[a]);
            minExt[a] = std::min(minExt[a], proj);
            maxExt[a] = std::max(maxExt[a], proj);
        }
    }

    obbCenter = Eigen::Vector3f::Zero();
    obbHalfExt = Eigen::Vector3f::Zero();
    for (int a = 0; a < 3; a++) {
        float mid = (minExt[a] + maxExt[a]) * 0.5f;
        obbHalfExt[a] = (maxExt[a] - minExt[a]) * 0.5f;
        obbCenter += mid * obbAxes[a];
    }

    std::cout << "[OBB] Axes:" << std::endl;
    for (int a = 0; a < 3; a++)
        std::cout << "  axis" << a << ": (" << obbAxes[a].x() << ", "
                  << obbAxes[a].y() << ", " << obbAxes[a].z() << ")" << std::endl;
    std::cout << "[OBB] Center: (" << obbCenter.x() << ", "
              << obbCenter.y() << ", " << obbCenter.z() << ")" << std::endl;
    std::cout << "[OBB] HalfExtents: (" << obbHalfExt.x() << ", "
              << obbHalfExt.y() << ", " << obbHalfExt.z() << ")" << std::endl;
}


void Map::initBoundary(std::vector<SamplePoint> &points) {
    this->xMin = FLT_MAX;
    this->xMax = FLT_MIN;
    this->yMin = FLT_MAX;
    this->yMax = FLT_MIN;
    this->zMin = FLT_MAX;
    this->zMax = FLT_MIN;
    for(SamplePoint& sp: points){
        xMin = std::min(sp.getPos()[0], xMin);
        xMax = std::max(sp.getPos()[0], xMax);
        yMin = std::min(sp.getPos()[1], yMin);
        yMax = std::max(sp.getPos()[1], yMax);
        zMin = std::min(sp.getPos()[2], zMin);
        zMax = std::max(sp.getPos()[2], zMax);
    }
    this->xOffset = 0 - this->xMin;
    this->yOffset = 0 - this->yMin;
}

void Map::initHeightMap(std::vector<SamplePoint> &points) {
    float mapWidthRange = this->xMax - this->xMin;
    float mapHeightRange = this->yMax - this->yMin;
    this->mapWidth = ceil(mapWidthRange / this->xResolution)+1;
    this->mapHeight = ceil(mapHeightRange / this->yResolution)+1;
    for(int i=0; i<this->mapWidth; i++){
        std::vector<float> xLine(this->mapHeight, this->zMin);
        this->heightMap.push_back(std::move(xLine));
    }
    // 更新heightMap
    for(SamplePoint& sp: points){
        int row = ceil((sp.getPos()[0] + this->xOffset) / this->xResolution);
        int col = ceil((sp.getPos()[1] + this->yOffset) / this->yResolution);
        this->heightMap[row][col] = std::max(sp.getPos()[2], this->heightMap[row][col]);
    }
}

float Map::getXOffset() {
    return this->xOffset;
}

float Map::getYOffset() {
    return this->yOffset;
}

int Map::getXResolution() {
    return this->xResolution;
}

int Map::getYResolution() {
    return this->yResolution;
}

int Map::getMapWidth() {
    return this->mapWidth;
}

int Map::getMapHeight() {
    return this->mapHeight;
}

float Map::getMapData(int row, int col) {
    return this->heightMap[row][col];
}

float Map::getMinHeightOfPos(float xPos, float yPos) {
    int row = ceil(xPos + xOffset) / xResolution;
    int col = ceil(yPos + yOffset) / yResolution;
    if(row >= 0 && row < mapWidth && col >= 0 && col < mapHeight){
        // 返回高度
        return heightMap[row][col];
    }
    return -1;
}