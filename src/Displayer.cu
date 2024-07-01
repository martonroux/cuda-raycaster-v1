/*
** RAYCASTING LIBRARY
** Displayer.cu
** Created by marton on 29/06/24.
*/

#include "Displayer.hpp"

namespace rcr {

    void Displayer::createImage(rcr::matrix3<rcr::hitPos> image) {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < height_; j++) {
            for (int i = 0; i < width_; i++) {
                auto* pixel = img_.ptr<uchar>(j, i);
                pixel[0] = image(j, i, 0, nullptr).hit ? 255 : 0;
                if (shapes_.size() > 1)
                    pixel[1] = image(j, i, 1, nullptr).hit ? 255 : 0;
                if (shapes_.size() > 2)
                    pixel[2] = image(j, i, 2, nullptr).hit ? 255 : 0;
            }
        }
    }

    void Displayer::displayImage() {
        keyboard_.resetPresses();
        mouse_.resetPresses();

        cv::imshow("Raycaster", img_);

        int delay = static_cast<int>(1.f / static_cast<float>(fps_) * 1000);

        if (delay == 0)
            delay = 1;
        int key = cv::waitKey(delay);

        keyboard_.setKeyPressed(static_cast<Keys>(key), true);
    }

    std::pair<int, int> Displayer::getNumThreadsBlocks() const
    {
        int numBlocks = (static_cast<int>(shapes_.size() * width_ * height_) + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

        return {numBlocks, NUM_THREADS_PER_BLOCK};
    }

    Triangle *Displayer::createTriangleArray() const
    {
        auto *h_triangles = (Triangle *)malloc(sizeof(Triangle) * shapes_.size());
        Triangle *d_triangles;

        for (int i = 0; i < static_cast<int>(shapes_.size()); i++) {
            memcpy(&h_triangles[i], &shapes_[i], sizeof(Triangle));
        }

        checkCudaError(cudaMalloc((void**)&d_triangles, sizeof(rcr::Triangle) * shapes_.size()), "cudaMalloc d_triangle");
        checkCudaError(cudaMemcpy(d_triangles, h_triangles, sizeof(rcr::Triangle) * shapes_.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_triangle");

        free(h_triangles);
        return d_triangles;
    }

    hitPos *Displayer::getDeviceHits() const {
        hitPos *d_hits;

        cudaMalloc((void**)&d_hits, sizeof(hitPos) * height_ * width_ * shapes_.size());
        return d_hits;
    }

    hitPos *Displayer::moveHitsToHost(hitPos *d_hits) const {
        auto *h_hits = (hitPos*)malloc(sizeof(hitPos) * height_ * width_ * shapes_.size());

        cudaMemcpy(h_hits, d_hits, sizeof(hitPos) * height_ * width_ * shapes_.size(), cudaMemcpyDeviceToHost);
        return h_hits;
    }

    Displayer::Displayer(size_t width, size_t height, size_t fps, rendererData data) : height_(height), width_(width), fps_(fps), screen_(data), img_(height, width, CV_8UC3, cv::Scalar(0, 0, 0))
    {
        cv::namedWindow("Raycaster");
        cv::setMouseCallback("Raycaster", rcr::onMouseCallback, &mouse_);
    }

    Displayer::~Displayer()
    {
        cv::destroyWindow("Raycaster");
    }

    void Displayer::addShape(Triangle triangle)
    {
        shapes_.push_back(triangle);
    }

    void Displayer::moveCamera(vec3<float> offset) {
        screen_.camPos.sumSingle(&offset, &screen_.camPos);
    }

    void Displayer::setCameraPos(vec3<float> pos) {
        screen_.camPos = pos;
    }

    void Displayer::render()
    {
        hitPos *d_hits = getDeviceHits();
        hitPos *h_hits;
        Triangle *d_triangles = createTriangleArray();
        CudaError *d_error = CudaError::createDeviceCudaError();
        std::pair<int, int> dimensions = getNumThreadsBlocks();

        kernelRender<<<dimensions.first, dimensions.second>>>(d_hits, height_, width_, shapes_.size(), d_triangles, screen_, d_error);

        checkCudaError(cudaGetLastError(), "kernel launch");
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        h_hits = moveHitsToHost(d_hits);

        // createImage(matrix3{height_, width_, shapes_.size(), h_hits});
        displayImage();

        rcr::CudaError::checkDeviceCudaError(d_error);

        free(h_hits);

        cudaFree(d_hits);
        cudaFree(d_triangles);
        cudaFree(d_error);
    }

    void Displayer::clear() {
        img_ = cv::Mat(static_cast<int>(height_), static_cast<int>(width_), CV_8UC3, cv::Scalar(0, 0, 0));
        shapes_.clear();
    }

    void Displayer::clear(rgb backgroundColor) {
        img_ = cv::Mat(static_cast<int>(height_), static_cast<int>(width_), CV_8UC3, cv::Scalar(backgroundColor.b, backgroundColor.g, backgroundColor.r));
        shapes_.clear();
    }

    Keyboard Displayer::getKeyboardFrame() const {
        return keyboard_;
    }

    Mouse Displayer::getMouseFrame() const {
        return mouse_;
    }

    vec3<float> Displayer::getCameraPos() const {
        return screen_.camPos;
    }
} // rcr
