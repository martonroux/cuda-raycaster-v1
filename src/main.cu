#include <iostream>
#include "CudaError.hpp"
#include "shapes/Triangle.hpp"
#include "render/Renderer.cuh"

#include <chrono>
#include <opencv2/opencv.hpp>

// Device 0: NVIDIA GeForce GTX 1080 Ti
// Max threads per block: 1024
// Max threads dimension (x, y, z): (1024, 1024, 64)
// Max grid size (x, y, z): (2147483647, 65535, 65535)

template<size_t H, size_t W>
__global__ void kernelRender(rcr::matrixh<H, W, rcr::hitPos> *image, rcr::Triangle *triangles, unsigned int nbTriangles, rcr::CudaError *error) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nbTriangles * H * W)
        return;
    rcr::screenData screen_data = {
        {-2, -2, 0},
        {4, 0, 0},
        {0, 4, 0}
    };
    rcr::rendererData data = {
        {0, 0, -4},
        screen_data
    };
    rcr::render<H, W>(image, triangles, nbTriangles, data, error);
}

void checkCudaError(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(result);
    }
}

rcr::Triangle *createTriangle() {
    rcr::vec3<float> p1 = {1, -2, 5};
    rcr::vec3<float> p2 = {0, 2, 5};
    rcr::vec3<float> p3 = {-4, -2, 5};
    rcr::Triangle h_triangle{p1, p2, p3};
    rcr::Triangle *d_triangle;

    checkCudaError(cudaMalloc((void**)&d_triangle, sizeof(rcr::Triangle)), "cudaMalloc d_triangle");
    checkCudaError(cudaMemcpy(d_triangle, &h_triangle, sizeof(rcr::Triangle), cudaMemcpyHostToDevice), "cudaMemcpy d_triangle");

    return d_triangle;
}

template<size_t H, size_t W>
void tempCreateImage(rcr::matrixh<H, W, rcr::hitPos> image) {
    cv::Mat temp(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            temp.at<cv::Vec3b>(j, i)[0] = image(j, i, nullptr).hit ? 255 : 0;
        }
    }
    cv::imshow("Raycaster", temp);
    cv::waitKey(0);
}

std::pair<int, int> getNumThreadsBlocks(size_t width, size_t height, unsigned int numThreadsPerBlock) {
    int numBlocks = (1 * width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;

    return {numBlocks, numThreadsPerBlock};
}

int main() {
    rcr::CudaError *d_error = rcr::CudaError::createDeviceCudaError();

    const size_t width = 1920;
    const size_t height = 1080;

    rcr::Triangle *d_triangles = createTriangle();
    rcr::matrixh<height, width, rcr::hitPos> h_image{};
    rcr::matrixh<height, width, rcr::hitPos> *d_image;

    h_image.moveToDevice();

    cudaMalloc((void**)&d_image, sizeof(rcr::matrixh<width, height, rcr::hitPos>));
    cudaMemcpy(d_image, &h_image, sizeof(rcr::matrixh<width, height, rcr::hitPos>), cudaMemcpyHostToDevice);

    std::pair<int, int> dimensions = getNumThreadsBlocks(width, height, 512);

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Blocks: " << dimensions.first << ", threads per block: " << dimensions.second << std::endl;
    kernelRender<height, width><<<dimensions.first, dimensions.second>>>(d_image, d_triangles, 1, d_error);

    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    rcr::CudaError::checkDeviceCudaError(d_error);

    cudaMemcpy(&h_image, d_image, sizeof(rcr::matrixh<width, height, rcr::hitPos>), cudaMemcpyDeviceToHost);
    h_image.moveToHost();

    cudaFree(d_triangles);
    cudaFree(d_error);

    tempCreateImage<height, width>(h_image);

    return 0;
}
