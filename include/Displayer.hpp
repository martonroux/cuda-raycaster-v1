/*
** RAYCASTING LIBRARY
** Displayer.hpp
** Created by marton on 29/06/24.
*/

#ifndef DISPLAYER_HPP
#define DISPLAYER_HPP

#define NUM_THREADS_PER_BLOCK 512

#include "shapes/Triangle.hpp"
#include "math/RGB.cuh"
#include "render/rendererData.h"
#include "render/Renderer.cuh"
#include "cudaHelpers.h"
#include "CudaError.hpp"
#include "inputs/Keyboard.hpp"
#include "inputs/Mouse.hpp"

#include <opencv2/opencv.hpp>

namespace rcr {

    class Displayer {
        size_t height_;
        size_t width_;
        size_t fps_;
        rendererData screen_;

        cv::Mat img_;
        std::vector<Triangle> shapes_{};

        Keyboard keyboard_{};
        Mouse mouse_{};

        void createImage(rcr::matrix3<rcr::hitPos> image);
        void displayImage();

        [[nodiscard]] std::pair<int, int> getNumThreadsBlocks() const;
        // [[nodiscard]] matrix3<rcr::hitPos> *createHitMatrix() const;
        // void retrieveDeviceMatrix(matrix3<rcr::hitPos> *d_matrix, size_t row, size_t col, size_t dep);
        [[nodiscard]] Triangle *createTriangleArray() const;
        [[nodiscard]] hitPos *getDeviceHits() const;
        [[nodiscard]] hitPos *moveHitsToHost(hitPos *d_hits) const;

    public:
        __host__ Displayer(size_t width, size_t height, size_t fps, rendererData data);
        __host__ ~Displayer();

        __host__ void addShape(Triangle triangle);
        __host__ void moveCamera(vec3<float> offset);
        __host__ void setCameraPos(vec3<float> pos);

        __host__ void render();
        __host__ void clear();
        __host__ void clear(rgb backgroundColor);

        __host__ [[nodiscard]] Keyboard getKeyboardFrame() const;
        __host__ [[nodiscard]] Mouse getMouseFrame() const;
        __host__ [[nodiscard]] vec3<float> getCameraPos() const;
    };

} // rcr

#endif //DISPLAYER_HPP
