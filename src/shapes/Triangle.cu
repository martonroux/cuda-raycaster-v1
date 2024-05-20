/*
** RAYCASTING LIBRARY
** Triangle.cu
** Created by marton on 20/05/24.
*/

#include "shapes/Triangle.hpp"

#include <cstdio>

namespace rcr {
    __host__ Triangle::Triangle(vec3<float> p1, vec3<float> p2, vec3<float> p3) {
        // Allocate memory in CUDA
        cudaMalloc((void**)&d_p1_, sizeof(rcr::vec3<float>));
        cudaMalloc((void**)&d_p2_, sizeof(rcr::vec3<float>));
        cudaMalloc((void**)&d_p3_, sizeof(rcr::vec3<float>));
//        cudaMalloc((void**)&d_mat_, sizeof(IMaterial));

        // Copy data to CUDA memory
        cudaMemcpy(d_p1_, &p1, sizeof(rcr::vec3<float>), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p2_, &p2, sizeof(rcr::vec3<float>), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p3_, &p3, sizeof(rcr::vec3<float>), cudaMemcpyHostToDevice);
//        cudaMemcpy(d_mat_, mat, sizeof(IMaterial), cudaMemcpyHostToDevice);
    }

    __host__ Triangle::~Triangle() {
        cudaFree(d_p1_);
        cudaFree(d_p2_);
        cudaFree(d_p3_);
//        cudaFree(d_mat_);
    }

    __device__ hitPos Triangle::hit(rcr::ray ray) const {
        __shared__ rcr::vec3<float> v0v1;
        __shared__ rcr::vec3<float> v1v2;

        ray.origin = {0, 0, -1};
        ray.direction = {0, 0, 1};
        __syncthreads();
        unsigned int tid = threadIdx.x;
        v0v1 = d_p2_->diff(d_p1_, tid);
        v1v2 = d_p3_->diff(d_p2_, tid);

        vec3<float> normal = v0v1.cross(&v1v2, tid);
        normal.normalize(tid);

        float nDotRayDirection = normal.dot(&ray.direction, tid);
        if (std::abs(nDotRayDirection) < 0.0000001)
            return {false, {}};

        float D = -normal.dot(d_p1_, tid);
        float t = -(normal.dot(&ray.origin, tid) + D) / normal.dot(&ray.direction, tid);

        if (t < 0)
            return {false, {}};
        vec3<float> vec1 = ray.direction.mult(t, tid); // dirMult
        vec3<float> P = ray.origin.sum(&vec1, tid);
        vec1 = P.diff(d_p1_, tid); // vp0
        vec3<float> C = v0v1.cross(&vec1, tid);

        if (normal.dot(&C, tid) < 0)
            return {false, {}};
        vec1 = P.diff(d_p2_, tid); // vp1
        C = v1v2.cross(&vec1, tid);

        if (normal.dot(&C, tid) < 0)
            return {false, {}};

        vec3<float> v2v0 = d_p1_->diff(d_p3_, tid);
        vec1 = P.diff(d_p3_, tid); // vp2
        C = v2v0.cross(&vec1, tid);

        if (normal.dot(&C, tid) < 0)
            return {false, {}};
        return {true, P};
    }

    __device__ float2 Triangle::getUvMapping(vec3<float> pos) const {
        return {0, 0};
    }

    __host__ void Triangle::changePosition(vec3<float> p1, vec3<float> p2, vec3<float> p3) {
        cudaMemcpy(d_p1_, &p1, sizeof(rcr::vec3<float>), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p2_, &p2, sizeof(rcr::vec3<float>), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p3_, &p3, sizeof(rcr::vec3<float>), cudaMemcpyHostToDevice);
    }

    __host__ void Triangle::changeMaterial(IMaterial *mat) {
//        cudaMemcpy(d_mat_, mat, sizeof(IMaterial), cudaMemcpyHostToDevice);
    }

}
