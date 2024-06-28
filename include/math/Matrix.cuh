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
    class matrix2 {
        T *h_values_ = nullptr;
        T *d_values_ = nullptr;

    public:
        __host__ matrix2();
        __host__ matrix2(T *values);
        __host__ ~matrix2();

        __host__ void moveToDevice();
        __host__ void moveToHost();

        __device__ __host__ T& operator()(size_t row, size_t col, CudaError *cuda_error);
        __device__ __host__ const T& operator()(size_t row, size_t col, CudaError *cuda_error) const;
    };

    template<size_t ROW, size_t COL, typename T>
    matrix2<ROW, COL, T>::matrix2() {
        h_values_ = (T*)malloc(sizeof(T) * ROW * COL);

        if (h_values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "Matrix.cuh | matrix2::matrix2");
    }

    template<size_t ROW, size_t COL, typename T>
    matrix2<ROW, COL, T>::matrix2(T *values) {
        h_values_ = (T*)malloc(sizeof(T) * ROW * COL);

        if (h_values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "Matrix.cuh | matrix2::matrix2");

        for (int i = 0; i < ROW * COL; i++)
            h_values_[i] = values[i];
    }

    template<size_t ROW, size_t COL, typename T>
    matrix2<ROW, COL, T>::~matrix2() {
    }

    template<size_t ROW, size_t COL, typename T>
    void matrix2<ROW, COL, T>::moveToDevice() {
        if (d_values_ != nullptr) {
            cudaFree(d_values_);
        }
        cudaError_t err = cudaMalloc((void**)&d_values_, sizeof(T) * ROW * COL);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)), "Matrix.cuh | matrix2::moveToDevice");

        err = cudaMemcpy(d_values_, h_values_, sizeof(T) * ROW * COL, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory copy to device failed: " + std::string(cudaGetErrorString(err)), "Matrix.cuh | matrix2::moveToDevice");
    }

    template<size_t ROW, size_t COL, typename T>
    void matrix2<ROW, COL, T>::moveToHost() {
        if (d_values_ == nullptr || h_values_ == nullptr)
            throw MatrixError("CUDA memory copy to host failed: null pointer", "Matrix.cuh | matrix2::moveToHost");

        free(h_values_);
        h_values_ = (T*)malloc(sizeof(T) * ROW * COL);

        cudaError_t err = cudaMemcpy(h_values_, d_values_, sizeof(T) * ROW * COL, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory copy to host failed: " + std::string(cudaGetErrorString(err)), "Matrix.cuh | matrix2::moveToHost");
    }

    template<size_t ROW, size_t COL, typename T>
    T & matrix2<ROW, COL, T>::operator()(size_t row, size_t col, CudaError *cuda_error) {
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
    const T& matrix2<ROW, COL, T>::operator()(size_t row, size_t col, CudaError *cuda_error) const {
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
