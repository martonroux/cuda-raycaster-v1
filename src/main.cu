#include <iostream>
#include "CudaError.hpp"
#include "shapes/Triangle.hpp"
#include "render/Renderer.cuh"
#include "math/Matrix2.cuh"
#include "math/Matrix3.cuh"

#include <chrono>
#include <opencv2/opencv.hpp>

// Device 0: NVIDIA GeForce GTX 1080 Ti
// Max threads per block: 1024
// Max threads dimension (x, y, z): (1024, 1024, 64)
// Max grid size (x, y, z): (2147483647, 65535, 65535)

template<size_t H, size_t W, size_t nTriangles>
__global__ void kernelRender(rcr::matrix3<H, W, nTriangles, rcr::hitPos> *image, rcr::Triangle *triangles, rcr::CudaError *error) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nTriangles * H * W)
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
    rcr::render<H, W, nTriangles>(image, triangles, data, error);
}

void checkCudaError(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(result);
    }
}

rcr::Triangle *createTriangles() {
    rcr::vec3<float> p1 = {1, -2, 5};
    rcr::vec3<float> p2 = {0, 2, 5};
    rcr::vec3<float> p3 = {-4, -2, 5};
    rcr::Triangle h_triangle1{p1, p2, p3};
    rcr::Triangle h_triangle2{rcr::vec3<float>{2, 2, 5}, p2, p3};
    rcr::Triangle h_triangles[2] = {h_triangle1, h_triangle2};
    rcr::Triangle *d_triangles;

    checkCudaError(cudaMalloc((void**)&d_triangles, sizeof(rcr::Triangle) * 2), "cudaMalloc d_triangle");
    checkCudaError(cudaMemcpy(d_triangles, &h_triangles, sizeof(rcr::Triangle) * 2, cudaMemcpyHostToDevice), "cudaMemcpy d_triangle");

    return d_triangles;
}

template<size_t H, size_t W, size_t nTriangles>
void tempCreateImage(rcr::matrix3<H, W, nTriangles, rcr::hitPos> image) {
    cv::Mat temp(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            temp.at<cv::Vec3b>(j, i)[0] = image(j, i, 0, nullptr).hit ? 255 : 0;
            temp.at<cv::Vec3b>(j, i)[1] = image(j, i, 1, nullptr).hit ? 255 : 0;
        }
    }
    cv::imshow("Raycaster", temp);
    cv::waitKey(0);
}

std::pair<int, int> getNumThreadsBlocks(size_t width, size_t height, size_t numTriangles, unsigned int numThreadsPerBlock) {
    int numBlocks = (numTriangles * width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;

    return {numBlocks, numThreadsPerBlock};
}

int main() {
    rcr::CudaError *d_error = rcr::CudaError::createDeviceCudaError();

    const size_t width = 1920;
    const size_t height = 1080;
    const size_t nTriangles = 2;

    rcr::Triangle *d_triangles = createTriangles();
    rcr::matrix3<height, width, nTriangles, rcr::hitPos> h_image{};
    rcr::matrix3<height, width, nTriangles, rcr::hitPos> *d_image;

    h_image.moveToDevice();

    cudaMalloc((void**)&d_image, sizeof(rcr::matrix3<width, height, nTriangles, rcr::hitPos>));
    cudaMemcpy(d_image, &h_image, sizeof(rcr::matrix3<width, height, nTriangles, rcr::hitPos>), cudaMemcpyHostToDevice);

    std::pair<int, int> dimensions = getNumThreadsBlocks(width, height, nTriangles, 512);

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Blocks: " << dimensions.first << ", threads per block: " << dimensions.second << std::endl;
    kernelRender<height, width, nTriangles><<<dimensions.first, dimensions.second>>>(d_image, d_triangles, d_error);

    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    rcr::CudaError::checkDeviceCudaError(d_error);

    cudaMemcpy(&h_image, d_image, sizeof(rcr::matrix2<width, height, rcr::hitPos>), cudaMemcpyDeviceToHost);
    h_image.moveToHost();

    cudaFree(d_triangles);
    cudaFree(d_error);

    tempCreateImage<height, width>(h_image);

    return 0;
}
