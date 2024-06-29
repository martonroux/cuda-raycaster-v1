/*
** RAYCASTING LIBRARY
** cudaHelpers.cu
** Created by marton on 29/06/24.
*/

#include "cudaHelpers.h"
#include <iostream>

void checkCudaError(cudaError_t result, const char *func)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(result);
    }
}
