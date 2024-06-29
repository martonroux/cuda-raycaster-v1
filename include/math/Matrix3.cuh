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
    template<typename T>
    class matrix3 {
        size_t row_;
        size_t col_;
        size_t dep_;
        
        T *h_values_ = nullptr;
        T *d_values_ = nullptr;

    public:
        __host__ matrix3(size_t rows, size_t cols, size_t depth);
        __host__ matrix3(size_t rows, size_t cols, size_t depth, T *values);
        __host__ ~matrix3() = default;

        __host__ void moveToDevice();
        __host__ void moveToHost();

        __device__ __host__ T& operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error);
        __device__ __host__ const T& operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error) const;
    };

    template<typename T>
    ::rcr::matrix3<T>::matrix3(size_t rows, size_t cols, size_t depth) : row_(rows), col_(cols), dep_(depth)
    {
        h_values_ = (T*)malloc(sizeof(T) * row_ * col_ * dep_);

        if (h_values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "Matrix3.cuh | matrix3::matrix3");
    }

    template<typename T>
    ::rcr::matrix3<T>::matrix3(size_t rows, size_t cols, size_t depth, T *values) : row_(rows), col_(cols), dep_(depth)
    {
        h_values_ = (T*)malloc(sizeof(T) * row_ * col_ * dep_);

        if (h_values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "Matrix3.cuh | matrix3::matrix3");

        for (int i = 0; i < row_ * col_ * dep_; i++)
            h_values_[i] = values[i];
    }

    template<typename T>
    void matrix3<T>::moveToDevice() {
        if (d_values_ != nullptr) {
            cudaFree(d_values_);
        }
        cudaError_t err = cudaMalloc((void**)&d_values_, sizeof(T) * row_ * col_ * dep_);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)), "Matrix3.cuh | matrix3::moveToDevice");

        err = cudaMemcpy(d_values_, h_values_, sizeof(T) * row_ * col_ * dep_, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory copy to device failed: " + std::string(cudaGetErrorString(err)), "Matrix3.cuh | matrix3::moveToDevice");
    }

    template<typename T>
    void matrix3<T>::moveToHost() {
        if (d_values_ == nullptr || h_values_ == nullptr)
            throw MatrixError("CUDA memory copy to host failed: null pointer", "Matrix3.cuh | matrix3::moveToHost");

        free(h_values_);
        h_values_ = (T*)malloc(sizeof(T) * row_ * col_ * dep_);

        cudaError_t err = cudaMemcpy(h_values_, d_values_, sizeof(T) * row_ * col_ * dep_, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
            throw MatrixError("CUDA memory copy to host failed: " + std::string(cudaGetErrorString(err)), "Matrix3.cuh | matrix3::moveToHost");
    }

    template<typename T>
    T & matrix3<T>::operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error) {
#ifdef __CUDA_ARCH__
        if (row >= row_ || col >= col_ || dep >= dep_) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator()");
            return d_values_[0];
        }
        return d_values_[dep + col * dep_ + row * col_ * dep_];
#else
        if (row >= row_ || col >= col_ || dep >= dep_)
            throw MatrixError("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator() const");
        return h_values_[dep + col * dep_ + row * col_ * dep_];
#endif
    }

    template<typename T>
    const T & matrix3<T>::operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error) const {
#ifdef __CUDA_ARCH__
        if (row >= row_ || col >= col_ || dep >= dep_) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator() const");
            return d_values_[0];
        }
        return d_values_[dep + col * dep_ + row * col_ * dep_];
#else
        if (row >= row_ || col >= col_ || dep >= dep_)
            throw MatrixError("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator() const");
        return h_values_[dep + col * dep_ + row * col_ * dep_];
#endif
    }

}

#endif //MATRIX3_CUH
