#include <iostream>
#include "CudaError.hpp"
#include "math/Ray.cuh"
#include "shapes/Triangle.hpp"

#include <chrono>
#include <math/Matrix.cuh>
#include <opencv2/opencv.hpp>
#include <render/Renderer.hpp>

// Device 0: NVIDIA GeForce GTX 1080 Ti
// Max threads per block: 1024
// Max threads dimension (x, y, z): (1024, 1024, 64)
// Max grid size (x, y, z): (2147483647, 65535, 65535)

// __global__ void triangleTest(rcr::Triangle *triangle, rcr::CudaError *cudaError) {
//     (void)cudaError;
//     rcr::hitPos pos;
//     __shared__ rcr::ray ray;
//
//     if (threadIdx.x == 0) {
//         ray = {{0, 0, -1}, {0, 0, 1}};
//     }
//     __syncthreads();
//
//     unsigned int tid = threadIdx.x;
//
//     if (tid < 1000) {
//         pos = triangle->hit(ray);
//         pos.pos.x += 1;
//         // printf("[%d] Found: %d, coords: %f %f %f\n", tid, pos.hit, pos.pos.x, pos.pos.y, pos.pos.z);
//     }
// }

__global__ void kernelRender(rcr::Triangle *triangles, unsigned int nbTriangles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nbTriangles * 4 * 4)
        return;
    rcr::render(nullptr, triangles, nbTriangles, 4, 4);
}

void checkCudaError(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(result);
    }
}

rcr::Triangle *createTriangle() {
    rcr::vec3<float> p1 = {2, 0, 0};
    rcr::vec3<float> p2 = {0, 2, 0};
    rcr::vec3<float> p3 = {-2, -2, 1};
    rcr::Triangle h_triangle{p1, p2, p3};
    rcr::Triangle *d_triangle;

    checkCudaError(cudaMalloc((void**)&d_triangle, sizeof(rcr::Triangle)), "cudaMalloc d_triangle");
    checkCudaError(cudaMemcpy(d_triangle, &h_triangle, sizeof(rcr::Triangle), cudaMemcpyHostToDevice), "cudaMemcpy d_triangle");

    return d_triangle;
}

int main() {
    rcr::CudaError h_error{};
    rcr::CudaError *d_error;

    cudaMalloc((void**)&d_error, sizeof(rcr::CudaError));
    cudaMemcpy(d_error, &h_error, sizeof(rcr::CudaError), cudaMemcpyHostToDevice);

    rcr::Triangle *d_triangles = createTriangle();

    int numThreadsPerBlock = 512;
    int numBlocks = (1 * 4 * 4 + numThreadsPerBlock - 1) / numThreadsPerBlock;
    std::cout << numThreadsPerBlock << " " << numBlocks << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    kernelRender<<<numBlocks, numThreadsPerBlock>>>(d_triangles, 1);

    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy error data back to host and check for errors
    checkCudaError(cudaMemcpy(&h_error, d_error, sizeof(rcr::CudaError), cudaMemcpyDeviceToHost), "cudaMemcpy d_error");

    try {
        h_error.throwException();
    } catch (const rcr::CudaException& e) {
        std::cerr << e.where() << ": " << e.what() << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    cudaFree(d_triangles);
    cudaFree(d_error);

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
