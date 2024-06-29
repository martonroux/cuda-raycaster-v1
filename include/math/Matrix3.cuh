/*
** RAYCASTING LIBRARY
** Matrix3.cuh
** Created by marton on 28/06/24.
*/

#ifndef MATRIX3_CUH
#define MATRIX3_CUH

#include "Error.hpp"
#include "CudaError.hpp"

namespace rcr {
    /* -------------------------- 3D MATRIX -------------------------- */
    template<size_t ROW, size_t COL, size_t DEP, typename T>
    class matrix3 {
        T *h_values_ = nullptr;
        T *d_values_ = nullptr;

    public:
        __host__ matrix3();
        __host__ matrix3(T *values);
        __host__ ~matrix3() = default;

        __host__ void moveToDevice();
        __host__ void moveToHost();

        __device__ __host__ T& operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error);
        __device__ __host__ const T& operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error) const;
    };

    template<size_t ROW, size_t COL, size_t DEP, typename T>
    matrix3<ROW, COL, DEP, T>::matrix3() {
        h_values_ = (T*)malloc(sizeof(T) * ROW * COL * DEP);

        if (h_values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "Matrix3.cuh | matrix3::matrix3");
    }

    template<size_t ROW, size_t COL, size_t DEP, typename T>
    matrix3<ROW, COL, DEP, T>::matrix3(T *values) {
        h_values_ = (T*)malloc(sizeof(T) * ROW * COL * DEP);

        if (h_values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "Matrix3.cuh | matrix3::matrix3");

        for (int i = 0; i < ROW * COL * DEP; i++)
            h_values_[i] = values[i];
    }

    template<size_t ROW, size_t COL, size_t DEP, typename T>
    void matrix3<ROW, COL, DEP, T>::moveToDevice() {
        if (d_values_ != nullptr) {
            cudaFree(d_values_);
        }
        cudaError_t err = cudaMalloc((void**)&d_values_, sizeof(T) * ROW * COL * DEP);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)), "Matrix3.cuh | matrix3::moveToDevice");

        err = cudaMemcpy(d_values_, h_values_, sizeof(T) * ROW * COL * DEP, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory copy to device failed: " + std::string(cudaGetErrorString(err)), "Matrix3.cuh | matrix3::moveToDevice");
    }

    template<size_t ROW, size_t COL, size_t DEP, typename T>
    void matrix3<ROW, COL, DEP, T>::moveToHost() {
        if (d_values_ == nullptr || h_values_ == nullptr)
            throw MatrixError("CUDA memory copy to host failed: null pointer", "Matrix3.cuh | matrix3::moveToHost");

        free(h_values_);
        h_values_ = (T*)malloc(sizeof(T) * ROW * COL * DEP);

        cudaError_t err = cudaMemcpy(h_values_, d_values_, sizeof(T) * ROW * COL * DEP, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory copy to host failed: " + std::string(cudaGetErrorString(err)), "Matrix3.cuh | matrix3::moveToHost");
    }

    template<size_t ROW, size_t COL, size_t DEP, typename T>
    T & matrix3<ROW, COL, DEP, T>::operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error) {
#ifdef __CUDA_ARCH__
        if (row >= ROW || col >= COL || dep >= DEP) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator()");
            return d_values_[0];
        }
        return d_values_[dep + col * DEP + row * COL * DEP];
#else
        if (row >= ROW || col >= COL || dep >= DEP)
            throw MatrixError("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator() const");
        return h_values_[dep + col * DEP + row * COL * DEP];
#endif
    }

    template<size_t ROW, size_t COL, size_t DEP, typename T>
    const T & matrix3<ROW, COL, DEP, T>::operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error) const {
#ifdef __CUDA_ARCH__
        if (row >= ROW || col >= COL || dep >= DEP) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator() const");
            return d_values_[0];
        }
        return d_values_[dep + col * DEP + row * COL * DEP];
#else
        if (row >= ROW || col >= COL || dep >= DEP)
            throw MatrixError("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator() const");
        return h_values_[dep + col * DEP + row * COL * DEP];
#endif
    }
}

#endif //MATRIX3_CUH
