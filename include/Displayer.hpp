/*
** RAYCASTING LIBRARY
** Displayer.hpp
** Created by marton on 29/06/24.
*/

#ifndef DISPLAYER_HPP
#define DISPLAYER_HPP

#include "shapes/Triangle.hpp"
#include "math/RGB.cuh"
#include "render/rendererData.h"
#include "render/Renderer.cuh"
#include "cudaHelpers.h"
#include "CudaError.hpp"

#include <opencv2/opencv.hpp>

namespace rcr {

    class Displayer {
        size_t height_;
        size_t width_;
        rendererData screen_;

        cv::Mat img_;
        std::vector<Triangle> shapes_{};

        std::pair<int, int> getNumThreadsBlocks(unsigned int numThreadsPerBlock) const;
        matrix3<rcr::hitPos> * createHitMatrix() const;
        static matrix3<rcr::hitPos> retrieveDeviceMatrix(matrix3<rcr::hitPos> * d_matrix, size_t row, size_t col, size_t dep);
        Triangle * createTriangleArray() const;

    public:
        __host__ Displayer(size_t width, size_t height, size_t fps, rendererData data);
        __host__ ~Displayer() = default;

        __host__ void addShape(Triangle triangle);
        __host__ void render();
        __host__ void clear();
        __host__ void clear(rgb backgroundColor);
    };

} // rcr

#endif //DISPLAYER_HPP
