/*
** RAYCASTING LIBRARY
** Displayer.cu
** Created by marton on 29/06/24.
*/

#include "Displayer.hpp"

namespace rcr {

    void Displayer::tempCreateImage(rcr::matrix3<rcr::hitPos> image) {
        keyboard_.resetPresses();
        mouse_.resetPresses();

        for (int i = 0; i < width_; i++) {
            for (int j = 0; j < height_; j++) {
                img_.at<cv::Vec3b>(j, i)[0] = image(j, i, 0, nullptr).hit ? 255 : 0;
                if (shapes_.size() > 1)
                    img_.at<cv::Vec3b>(j, i)[1] = image(j, i, 1, nullptr).hit ? 255 : 0;
                if (shapes_.size() > 2)
                    img_.at<cv::Vec3b>(j, i)[2] = image(j, i, 2, nullptr).hit ? 255 : 0;
            }
        }
        cv::imshow("Raycaster", img_);
        int key = cv::waitKey(static_cast<int>(1.f / static_cast<float>(fps_) * 1000));

        keyboard_.setKeyPressed(static_cast<Keys>(key), true);
    }

    std::pair<int, int> Displayer::getNumThreadsBlocks() const
    {
        int numBlocks = (static_cast<int>(shapes_.size() * width_ * height_) + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

        return {numBlocks, NUM_THREADS_PER_BLOCK};
    }

    // matrix3<rcr::hitPos> *Displayer::createHitMatrix() const
    // {
    //     rcr::hitPos *d_values;
    //
    //     cudaMalloc((void**)&d_values, sizeof(rcr::hitPos) * height_ * width_ * shapes_.size());
    //
    //     rcr::matrix3<rcr::hitPos> h_image{height_, width_, shapes_.size(), d_values};
    //     rcr::matrix3<rcr::hitPos> *d_image;
    //
    //     cudaMalloc((void **) &d_image, sizeof(rcr::matrix3<rcr::hitPos>));
    //     cudaMemcpy(d_image, &h_image, sizeof(rcr::matrix3<rcr::hitPos>), cudaMemcpyHostToDevice);
    //
    //     return d_image;
    // }
    //
    // matrix3<rcr::hitPos> Displayer::retrieveDeviceMatrix(matrix3<rcr::hitPos> *d_matrix, size_t row, size_t col,
    //                                                      size_t dep)
    // {
    //     auto *h_matrix = (matrix3<rcr::hitPos>*)malloc(sizeof(matrix3<rcr::hitPos>));
    //     auto *h_values = (rcr::hitPos*)malloc(sizeof(rcr::hitPos) * row * col * dep);
    //
    //     cudaMemcpy(h_matrix, d_matrix, sizeof(rcr::matrix3<rcr::hitPos>), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_values, h_matrix->getValues(), sizeof(rcr::hitPos) * row * col * dep, cudaMemcpyDeviceToHost);
    //
    //     return matrix3{row, col, dep, h_values};
    // }

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

    void Displayer::render()
    {
        hitPos *d_hits = getDeviceHits();
        hitPos *h_hits;
        Triangle *d_triangles = createTriangleArray();
        CudaError *d_error = CudaError::createDeviceCudaError();
        std::pair<int, int> dimensions = getNumThreadsBlocks();

        kernelRender<<<dimensions.first, dimensions.second>>>(d_hits, height_, width_, shapes_.size(), d_triangles, screen_, d_error);

        h_hits = moveHitsToHost(d_hits);

        // tempCreateImage(matrix3{height_, width_, shapes_.size(), h_hits});

        free(h_hits);

        cudaFree(d_hits);
        cudaFree(d_triangles);
        cudaFree(d_error);
    }

    void Displayer::clear() {
        img_ = cv::Mat(static_cast<int>(height_), static_cast<int>(width_), CV_8UC3, cv::Scalar(0, 0, 0));
    }

    void Displayer::clear(rgb backgroundColor) {
        img_ = cv::Mat(static_cast<int>(height_), static_cast<int>(width_), CV_8UC3, cv::Scalar(backgroundColor.b, backgroundColor.g, backgroundColor.r));
    }

    Keyboard Displayer::getKeyboardFrame() const {
        return keyboard_;
    }

    Mouse Displayer::getMouseFrame() const {
        return mouse_;
    }
} // rcr
