/*
** RAYCASTING LIBRARY
** Renderer.hpp
** Created by marton on 15/06/24.
*/

#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <vector>
#include "shapes/Triangle.hpp"
#include "CudaError.hpp"
#include <opencv2/opencv.hpp>

namespace rcr {

    // class Renderer {
    //     unsigned int height_;
    //     unsigned int width_;
    //
    // public:
    //     __host__ Renderer(unsigned int width, unsigned int height) : height_(height), width_(width) {}
    //     __host__ ~Renderer() = default;
    //
    //     // RUNTIME
    //     __device__ void render(cv::Mat *image, Triangle *triangles, unsigned int nbTriangles);
    // };

    inline __device__ void render(cv::Mat *image, Triangle *triangles, unsigned int nbTriangles, unsigned int width, unsigned int height) {
        (void)image;
        (void)triangles;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int nbElems = static_cast<int>(nbTriangles);
        int triangleId = idx % nbElems;

        printf("Thread %d was assigned triangle %d at pixel (%d,%d) (%d %d)\n", idx, triangleId, idx % width, idx / width, idx, width);
    }

} // rcr

#endif //RENDERER_HPP
