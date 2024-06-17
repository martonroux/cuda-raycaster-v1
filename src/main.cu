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
__global__ void kernelRender(rcr::matrixh<H, W, rcr::hitPos> *image, rcr::Triangle *triangles, unsigned int nbTriangles, rcr::CudaError *error) {
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
void tempCreateImage(rcr::matrixh<H, W, rcr::hitPos> image) {
    cv::Mat temp(W, H, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            temp.at<cv::Vec3b>(i, j)[0] = image(j, i).hit ? 255 : 0;
        }
    }
    cv::imshow("Raycaster", temp);
    cv::waitKey(0);
}

// __global__ void kernel(rcr::matrixh<2, 2, rcr::hitPos> *matrix) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < 4) {
//         int row = idx / 2;
//         int col = idx % 2;
//         rcr::hitPos pos = (*matrix)(row, col);
//         (*matrix)(row, col).hit = false;
//         printf("hit: %d, x: %f, y: %f, z: %f\n", pos.hit, pos.pos.x, pos.pos.y, pos.pos.z);
//     }
// }
//
// int main() {
//     // Initialize the matrix with values
//     rcr::hitPos values[4] = {
//         {true, 1.0f, 2.0f, 3.0f}, {false, 4.0f, 5.0f, 6.0f},
//         {true, 7.0f, 8.0f, 9.0f}, {false, 10.0f, 11.0f, 12.0f}
//     };
//
//     rcr::matrixh<2, 2, rcr::hitPos> matrix(values);
//     rcr::matrixh<2, 2, rcr::hitPos> *d_matrix;
//
//     // Access elements on host
//     for (size_t row = 0; row < 2; ++row) {
//         for (size_t col = 0; col < 2; ++col) {
//             rcr::hitPos pos = matrix(row, col);
//             std::cout << "Host access - hit: " << pos.hit << ", x: " << pos.pos.x << ", y: " << pos.pos.y << ", z: " << pos.pos.z << std::endl;
//         }
//     }
//
//     // Move data to device
//     matrix.moveToDevice();
//
//     cudaMalloc((void**)&d_matrix, sizeof(rcr::matrixh<2, 2, rcr::hitPos>));
//     cudaMemcpy(d_matrix, &matrix, sizeof(rcr::matrixh<2, 2, rcr::hitPos>), cudaMemcpyHostToDevice);
//
//     // Launch kernel
//     kernel<<<1, 4>>>(d_matrix);
//     cudaDeviceSynchronize();
//
//     cudaMemcpy(&matrix, d_matrix, sizeof(rcr::matrixh<2, 2, rcr::hitPos>), cudaMemcpyDeviceToHost);
//     // Move data back to host
//     // matrix.moveToHost();
//     //
//     // Verify data moved back correctly (if you modified data in the kernel, you should verify it here)
//     matrix.moveToHost();
//     for (size_t row = 0; row < 2; ++row) {
//         for (size_t col = 0; col < 2; ++col) {
//             rcr::hitPos pos = matrix(row, col);
//             std::cout << "Host verification - hit: " << pos.hit << ", x: " << pos.pos.x << ", y: " << pos.pos.y << ", z: " << pos.pos.z << std::endl;
//         }
//     }
//
//     return 0;
// }

std::pair<int, int> getNumThreadsBlocks(size_t width, size_t height, unsigned int numThreadsPerBlock) {
    int numBlocks = (1 * width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;

    return {numBlocks, numThreadsPerBlock};
}

int main() {
    rcr::CudaError h_error{};
    rcr::CudaError *d_error;
    cudaMalloc((void**)&d_error, sizeof(rcr::CudaError));
    cudaMemcpy(d_error, &h_error, sizeof(rcr::CudaError), cudaMemcpyHostToDevice);

    const size_t width = 512;
    const size_t height = 512;

    rcr::Triangle *d_triangles = createTriangle();
    rcr::matrixh<height, width, rcr::hitPos> h_image{};
    rcr::matrixh<height, width, rcr::hitPos> *d_image;

    h_image.moveToDevice();

    cudaMalloc((void**)&d_image, sizeof(rcr::matrixh<2, 2, rcr::hitPos>));
    cudaMemcpy(d_image, &h_image, sizeof(rcr::matrixh<2, 2, rcr::hitPos>), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    std::pair<int, int> dimensions = getNumThreadsBlocks(width, height, 512);

    kernelRender<height, width><<<dimensions.first, dimensions.second>>>(d_image, d_triangles, 1, d_error);

    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    checkCudaError(cudaMemcpy(&h_error, d_error, sizeof(rcr::CudaError), cudaMemcpyDeviceToHost), "cudaMemcpy d_error");

    try {
        h_error.throwException();
    } catch (const rcr::CudaException& e) {
        std::cerr << e.where() << ": " << e.what() << std::endl;
    }

    cudaMemcpy(&h_image, d_image, sizeof(rcr::matrixh<2, 2, rcr::hitPos>), cudaMemcpyDeviceToHost);
    h_image.moveToHost();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    cudaFree(d_triangles);
    cudaFree(d_error);

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
