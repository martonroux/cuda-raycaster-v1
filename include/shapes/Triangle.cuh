/*
** RAYCASTING LIBRARY
** Triangle.cuh
** Created by marton on 20/05/24.
*/

#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <cstdio>

#include "math/Ray.hpp"
#include "math/Vec3.hpp"

namespace rcr {

    struct TriangleProperties {
        vec3<float> p1;
        vec3<float> p2;
        vec3<float> p3;
    };

    inline __device__ rcr::hitPos triangleHit(void *shapeData, rcr::ray ray) {

    }

    // inline __device__ rcr::hitPos triangleHit(rcr::TriangleProperties *shapeData, rcr::ray ray) {
    //     __shared__ rcr::vec3<float> v0v1;
    //     __shared__ rcr::vec3<float> v1v2;
    //
    //     ray.origin = {0, 0, -1};
    //     ray.direction = {0, 0, 1};
    //     __syncthreads();
    //     unsigned int tid = threadIdx.x;
    //     v0v1 = shapeData->p2.diff(&shapeData->p1, tid);
    //     v1v2 = shapeData->p3.diff(&shapeData->p2, tid + 3);
    //
    //     printf("%d\n", tid);
    //     __syncthreads();
    //     rcr::vec3<float> normal = v0v1.cross(&v1v2, tid);
    //     normal.normalize(tid);
    //
    //     float nDotRayDirection = normal.dot(&ray.direction, tid);
    //     if (std::abs(nDotRayDirection) < 0.0000001)
    //         return {false, {}};
    //
    //     float D = -normal.dot(&shapeData->p1, tid);
    //     float t = -(normal.dot(&ray.origin, tid) + D) / normal.dot(&ray.direction, tid);
    //
    //     if (t < 0)
    //         return {false, {}};
    //     rcr::vec3<float> vec1 = ray.direction.mult(t, tid); // dirMult
    //     rcr::vec3<float> P = ray.origin.sum(&vec1, tid);
    //     vec1 = P.diff(&shapeData->p1, tid); // vp0
    //     rcr::vec3<float> C = v0v1.cross(&vec1, tid);
    //
    //     if (normal.dot(&C, tid) < 0)
    //         return {false, {}};
    //     vec1 = P.diff(&shapeData->p2, tid); // vp1
    //     C = v1v2.cross(&vec1, tid);
    //
    //     if (normal.dot(&C, tid) < 0)
    //         return {false, {}};
    //
    //     rcr::vec3<float> v2v0 = shapeData->p1.diff(&shapeData->p3, tid);
    //     vec1 = P.diff(&shapeData->p3, tid); // vp2
    //     C = v2v0.cross(&vec1, tid);
    //
    //     if (normal.dot(&C, tid) < 0)
    //         return {false, {}};
    //     return {true, P};
    // }

} // rcr

#endif //TRIANGLE_HPP
