#include <iostream>
#include "CudaError.hpp"
#include "shapes/Triangle.hpp"

#include <chrono>
#include <opencv2/opencv.hpp>
#include <render/Renderer.cuh>

// Device 0: NVIDIA GeForce GTX 1080 Ti
// Max threads per block: 1024
// Max threads dimension (x, y, z): (1024, 1024, 64)
// Max grid size (x, y, z): (2147483647, 65535, 65535)

template<size_t H, size_t W>
__global__ void kernelRender(rcr::matrix<H, W, rcr::hitPos> *image, rcr::Triangle *triangles, unsigned int nbTriangles, rcr::CudaError *error) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nbTriangles * H * W)
        return;
    rcr::screenData screen_data = {
        {-2, -1, 0},
        {4, 0, 0},
        {0, 2, 0}
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
void tempCreateImage(rcr::matrix<H, W, rcr::hitPos> image) {
    cv::Mat temp(W, H, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            temp.at<cv::Vec3b>(i, j)[0] = image(j, i).hit ? 255 : 0;
        }
    }
    cv::imshow("Modified Image", temp);
    cv::waitKey(0);
}

int main() {
    rcr::CudaError h_error{};
    rcr::CudaError *d_error;

    const size_t width = 200;
    const size_t height = 200;

    rcr::hitPos hits[width * height] = {};

    rcr::matrix<height, width, rcr::hitPos> h_image{hits};
    rcr::matrix<height, width, rcr::hitPos> *d_image;
    cudaMalloc((void**)&d_error, sizeof(rcr::CudaError));
    cudaMemcpy(d_error, &h_error, sizeof(rcr::CudaError), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_image, sizeof(rcr::matrix<height, width, rcr::hitPos>));
    cudaMemcpy(d_image, &h_image, sizeof(rcr::matrix<height, width, rcr::hitPos>), cudaMemcpyHostToDevice);

    rcr::Triangle *d_triangles = createTriangle();

    int numThreadsPerBlock = 512;
    int numBlocks = (1 * width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;
    std::cout << numThreadsPerBlock << " " << numBlocks << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    kernelRender<height, width><<<numBlocks, numThreadsPerBlock>>>(d_image, d_triangles, 1, d_error);

    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    checkCudaError(cudaMemcpy(&h_error, d_error, sizeof(rcr::CudaError), cudaMemcpyDeviceToHost), "cudaMemcpy d_error");

    try {
        h_error.throwException();
    } catch (const rcr::CudaException& e) {
        std::cerr << e.where() << ": " << e.what() << std::endl;
    }

    cudaMemcpy(&h_image, d_image, sizeof(rcr::matrix<height, width, rcr::hitPos>), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    cudaFree(d_triangles);
    cudaFree(d_error);
    cudaFree(d_image);

    tempCreateImage<height, width>(h_image);

    return 0;
}
//
// int main() {
//     rcr::CudaError h_error{};
//     rcr::CudaError *d_error;
//
//     cudaMalloc((void**)&d_error, sizeof(rcr::CudaError));
//     cudaMemcpy(d_error, &h_error, sizeof(rcr::CudaError), cudaMemcpyHostToDevice);
//
//     rcr::vec3<float> p1 = {2, 0, 0};
//     rcr::vec3<float> p2 = {0, 2, 0};
//     rcr::vec3<float> p3 = {-2, -2, 1};
//     rcr::Triangle h_triangle{p1, p2, p3};
//     rcr::Triangle *d_triangle;
//
//     checkCudaError(cudaMalloc((void**)&d_triangle, sizeof(rcr::Triangle)), "cudaMalloc d_triangle");
//
//     checkCudaError(cudaMemcpy(d_triangle, &h_triangle, sizeof(rcr::Triangle), cudaMemcpyHostToDevice), "cudaMemcpy d_triangle");
//
//     auto start = std::chrono::high_resolution_clock::now();
//
//     triangleTest<<<1, 512>>>(d_triangle, d_error);
//
//     checkCudaError(cudaGetLastError(), "kernel launch");
//     checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
//
//     // Copy error data back to host and check for errors
//     checkCudaError(cudaMemcpy(&h_error, d_error, sizeof(rcr::CudaError), cudaMemcpyDeviceToHost), "cudaMemcpy d_error");
//
//     try {
//         h_error.throwException();
//     } catch (const rcr::CudaException& e) {
//         std::cerr << e.where() << ": " << e.what() << std::endl;
//     }
//
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration = end - start;
//     std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
//
//     cudaFree(d_triangle);
//     cudaFree(d_error);
//
//     return 0;
// }

// int main() {
//     rcr::CudaError h_error{};
//     rcr::CudaError *d_error;
//
//     cudaMalloc((void**)&d_error, sizeof(rcr::CudaError));
//     cudaMemcpy(d_error, &h_error, sizeof(rcr::CudaError), cudaMemcpyHostToDevice);
//
//     float values[] = {1, 2, 3, 4};
//
//     rcr::Matrix<2, 2, float> h_matrix = rcr::Matrix<2, 2, float>(values);
//     rcr::Matrix<2, 2, float> *d_matrix;
//
//     checkCudaError(cudaMalloc((void**)&d_matrix, sizeof(rcr::Matrix<2, 2, float>)), "cudaMalloc d_matrix");
//
//     checkCudaError(cudaMemcpy(d_matrix, &h_matrix, sizeof(rcr::Matrix<2, 2, float>), cudaMemcpyHostToDevice), "cudaMemcpy d_matrix");
//
//     auto start = std::chrono::high_resolution_clock::now();
//
//     matrixTest<<<1, 1>>>(d_matrix);
//
//     checkCudaError(cudaGetLastError(), "kernel launch");
//     checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
//
//     // Copy error data back to host and check for errors
//     checkCudaError(cudaMemcpy(&h_error, d_error, sizeof(rcr::CudaError), cudaMemcpyDeviceToHost), "cudaMemcpy d_error");
//
//     try {
//         h_error.throwException();
//     } catch (const rcr::CudaException& e) {
//         std::cerr << e.where() << ": " << e.what() << std::endl;
//     }
//
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration = end - start;
//     std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
//
//     cudaFree(d_matrix);
//     cudaFree(d_error);
//
//     return 0;
// }
//
// int main() {
//     cv::Mat image(512, 512, CV_8UC3, cv::Scalar(0, 0, 255)); // Initialize to black
//
//     cv::imshow("Modified Image", image);
//     cv::waitKey(0);
//     return 0;
// }
