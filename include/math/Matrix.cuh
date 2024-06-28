/*
** RAYCASTING LIBRARY
** Matrix.cuh
** Created by marton on 15/06/24.
*/

#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <iostream>
#include "Error.hpp"

namespace rcr {

    /* -------------------------- 2D MATRIX -------------------------- */
    template<size_t ROW, size_t COL, typename T>
    class matrixh {
        T *h_values_ = nullptr;
        T *d_values_ = nullptr;

    public:
        __host__ matrixh();
        __host__ matrixh(T *values);
        __host__ ~matrixh();

        __host__ void moveToDevice();
        __host__ void moveToHost();

        __device__ __host__ T& operator()(size_t row, size_t col, CudaError *cuda_error);
        __device__ __host__ const T& operator()(size_t row, size_t col, CudaError *cuda_error) const;
    };

    template<size_t ROW, size_t COL, typename T>
    matrixh<ROW, COL, T>::matrixh() {
        h_values_ = (T*)malloc(sizeof(T) * ROW * COL);

        if (h_values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "Matrix.cuh | matrixh::matrixh");
    }

    template<size_t ROW, size_t COL, typename T>
    matrixh<ROW, COL, T>::matrixh(T *values) {
        h_values_ = (T*)malloc(sizeof(T) * ROW * COL);

        if (h_values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "Matrix.cuh | matrixh::matrixh");

        for (int i = 0; i < ROW * COL; i++)
            h_values_[i] = values[i];
    }

    template<size_t ROW, size_t COL, typename T>
    matrixh<ROW, COL, T>::~matrixh() {
    }

    template<size_t ROW, size_t COL, typename T>
    void matrixh<ROW, COL, T>::moveToDevice() {
        if (d_values_ != nullptr) {
            cudaFree(d_values_);
        }
        cudaError_t err = cudaMalloc((void**)&d_values_, sizeof(T) * ROW * COL);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)), "Matrix.cuh | matrixh::moveToDevice");

        err = cudaMemcpy(d_values_, h_values_, sizeof(T) * ROW * COL, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory copy to device failed: " + std::string(cudaGetErrorString(err)), "Matrix.cuh | matrixh::moveToDevice");
    }

    template<size_t ROW, size_t COL, typename T>
    void matrixh<ROW, COL, T>::moveToHost() {
        if (d_values_ == nullptr || h_values_ == nullptr)
            throw MatrixError("CUDA memory copy to host failed: null pointer", "Matrix.cuh | matrixh::moveToHost");

        free(h_values_);
        h_values_ = (T*)malloc(sizeof(T) * ROW * COL);

        cudaError_t err = cudaMemcpy(h_values_, d_values_, sizeof(T) * ROW * COL, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory copy to host failed: " + std::string(cudaGetErrorString(err)), "Matrix.cuh | matrixh::moveToHost");
    }

    template<size_t ROW, size_t COL, typename T>
    T & matrixh<ROW, COL, T>::operator()(size_t row, size_t col, CudaError *cuda_error) {
#ifdef __CUDA_ARCH__
        if (row > ROW || col > COL) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "Matrix.cuh | operator()");
            return d_values_[0];
        }
        return d_values_[col + row * COL];
#else
        return h_values_[col + row * COL];
#endif
    }

    template<size_t ROW, size_t COL, typename T>
    const T& matrixh<ROW, COL, T>::operator()(size_t row, size_t col, CudaError *cuda_error) const {
#ifdef __CUDA_ARCH__
        if (row > ROW || col > COL) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "Matrix.cuh | operator() const");
            return d_values_[0];
        }
        return d_values_[col + row * COL];
#else
        return h_values_[col + row * COL];
#endif
    }
}

#endif //MATRIX_CUH
