/*
** RAYCASTING LIBRARY
** main.cu
** Created by marton on 20/05/24.
*/

#include <iostream>
#include "CudaError.hpp"
#include "math/Ray.hpp"
#include "IShape.hpp"
#include "shapes/Triangle.hpp"
#include "math/Ray.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include "shapes/Triangle.hpp"
#include <chrono>

__global__ void triangleTest(rcr::Triangle* triangle) {
    rcr::hitPos pos{};
    __shared__ rcr::ray ray;

    ray = {{0, 0, -1}, {0, 0, 1}};
    for (int i = 0; i < 10; i++) {
        pos = triangle->hit(ray);

        if (threadIdx.x == 0)
            printf("Found: %d, coords: %f %f %f\n", pos.hit, pos.pos.x, pos.pos.y, pos.pos.z);
        __syncthreads();
    }
}

void checkCudaError(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(result);
    }
}

int main() {
    rcr::Triangle h_triangle({2, 0, 0}, {0, 2, 0}, {-2, -2, -1});
    rcr::Triangle* d_triangle;

    cudaMalloc((void**)&d_triangle, sizeof(rcr::Triangle));
    cudaMemcpy(d_triangle, &h_triangle, sizeof(rcr::Triangle), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    triangleTest<<<1, 3>>>(d_triangle);

    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    cudaFree(d_triangle);

    return 0;
}
