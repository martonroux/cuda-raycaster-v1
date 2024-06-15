#include <iostream>
#include "CudaError.hpp"
#include "math/Ray.hpp"
#include "shapes/Triangle.hpp"

#include <chrono>

// Device 0: NVIDIA GeForce GTX 1080 Ti
// Max threads per block: 1024
// Max threads dimension (x, y, z): (1024, 1024, 64)
// Max grid size (x, y, z): (2147483647, 65535, 65535)

__global__ void triangleTest(rcr::Triangle *triangle, rcr::CudaError *cudaError) {
    (void)cudaError;
    rcr::hitPos pos;
    __shared__ rcr::ray ray;

    if (threadIdx.x == 0) {
        ray = {{0, 0, -1}, {0, 0, 1}};
    }
    __syncthreads();

    unsigned int tid = threadIdx.x;

    if (tid < 1000) {
        pos = triangle->hit(ray);
        printf("[%d] Found: %d, coords: %f %f %f\n", tid, pos.hit, pos.pos.x, pos.pos.y, pos.pos.z);
    }
}

void checkCudaError(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(result);
    }
}

int main() {
    rcr::CudaError h_error{};
    rcr::CudaError *d_error;

    cudaMalloc((void**)&d_error, sizeof(rcr::CudaError));
    cudaMemcpy(d_error, &h_error, sizeof(rcr::CudaError), cudaMemcpyHostToDevice);

    rcr::vec3<float> p1 = {2, 0, 0};
    rcr::vec3<float> p2 = {0, 2, 0};
    rcr::vec3<float> p3 = {-2, -2, 1};
    rcr::Triangle h_triangle{p1, p2, p3};
    rcr::Triangle *d_triangle;

    checkCudaError(cudaMalloc((void**)&d_triangle, sizeof(rcr::Triangle)), "cudaMalloc d_triangle");

    checkCudaError(cudaMemcpy(d_triangle, &h_triangle, sizeof(rcr::Triangle), cudaMemcpyHostToDevice), "cudaMemcpy d_triangle");

    auto start = std::chrono::high_resolution_clock::now();

    triangleTest<<<1, 512>>>(d_triangle, d_error);

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

    cudaFree(d_triangle);
    cudaFree(d_error);

    return 0;
}
