/*
** RAYCASTING LIBRARY
** Displayer.cu
** Created by marton on 29/06/24.
*/

#include "Displayer.hpp"

namespace rcr {

    void Displayer::createImage(matrix2<rgb> image) {
        rgb *data = image.getValues();

        #pragma omp parallel for collapse(2)
        for (int j = 0; j < height_; j++) {
            for (int i = 0; i < width_; i++) {
                auto* pixel = img_.ptr<uchar>(j, i);
                memcpy(pixel, &data[j * width_ + i], sizeof(uchar) * 3);
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

    std::pair<int, int> Displayer::getNumThreadsBlocks(size_t numTriangles) const
    {
        int numBlocks = (static_cast<int>(numTriangles * width_ * height_) + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

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

    rgb *Displayer::getDeviceImage() const {
        rgb *d_hits;

        cudaMalloc((void**)&d_hits, sizeof(rgb) * height_ * width_);
        return d_hits;
    }

    rgb *Displayer::moveImageToHost(rgb *d_image) const {
        auto *h_image = (rgb*)malloc(sizeof(rgb) * height_ * width_);

        cudaMemcpy(h_image, d_image, sizeof(rgb) * height_ * width_, cudaMemcpyDeviceToHost);
        return h_image;
    }

    Displayer::Displayer(size_t width, size_t height, size_t fps, rendererData data) : height_(height), width_(width), fps_(fps), screen_(data), img_(height, width, CV_8UC3, cv::Scalar(0, 0, 0))
    {
        cv::namedWindow("Raycaster");
        cv::setMouseCallback("Raycaster", rcr::onMouseCallback, &mouse_);
        d_img_ = getDeviceImage();
    }

    Displayer::~Displayer()
    {
        cudaFree(d_img_);
        if (d_hits_ != nullptr) cudaFree(d_hits_);
        cv::destroyWindow("Raycaster");
    }

    void Displayer::addShape(Triangle triangle)
    {
        shapes_.push_back(triangle);
    }

    void Displayer::moveCamera(vec3<float> offset) {
        screen_.camPos.sum(&offset, &screen_.camPos);
    }

    void Displayer::setCameraPos(vec3<float> pos) {
        screen_.camPos = pos;
    }

    void Displayer::render()
    {
        if (prev_num_triangles_ != shapes_.size()) {
            if (d_hits_ != nullptr) cudaFree(d_hits_);
            d_hits_ = getDeviceHits();
        }
        Triangle *d_triangles = createTriangleArray();
        CudaError *d_error = CudaError::createDeviceCudaError();
        std::pair<int, int> dimensions = getNumThreadsBlocks(shapes_.size());
        rgb *h_image;

        kernelHitdetect<<<dimensions.first, dimensions.second>>>(d_hits_, height_, width_, shapes_.size(), d_triangles, screen_, d_error);

        checkCudaError(cudaGetLastError(), "Hit Detect kernel launch");
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize for Hit Detect kernel");
        rcr::CudaError::checkDeviceCudaError(d_error);

        dimensions = getNumThreadsBlocks(1);

        kernelRender<<<dimensions.first, dimensions.second>>>(d_img_, d_hits_, height_, width_, shapes_.size(), d_triangles, screen_, d_error);

        checkCudaError(cudaGetLastError(), "Render kernel launch");
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize for Render kernel");
        rcr::CudaError::checkDeviceCudaError(d_error);

        h_image = moveImageToHost(d_img_);

        createImage(matrix2{height_, width_, h_image});
        displayImage();

        free(h_image);
        cudaFree(d_triangles);
        cudaFree(d_error);

        prev_num_triangles_ = shapes_.size();
    }

    void Displayer::clear() {
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
