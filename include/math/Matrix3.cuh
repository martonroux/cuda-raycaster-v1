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
        
        T *values_ = nullptr;

    public:
        __host__ __device__ matrix3(size_t rows, size_t cols, size_t depth);
        __host__ __device__ matrix3(size_t rows, size_t cols, size_t depth, T *values);
        __host__ __device__ ~matrix3() = default;

        __host__ __device__ [[nodiscard]] T *getValues() const { return values_; }

        __host__ __device__ T& operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error);
        __host__ __device__ const T& operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error) const;
    };

    template<typename T>
    __host__ __device__ ::rcr::matrix3<T>::matrix3(size_t rows, size_t cols, size_t depth) : row_(rows), col_(cols), dep_(depth)
    {
        values_ = (T*)malloc(sizeof(T) * row_ * col_ * dep_);

        if (values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "Matrix3.cuh | matrix3::matrix3");
    }

    template<typename T>
    __host__ __device__ ::rcr::matrix3<T>::matrix3(size_t rows, size_t cols, size_t depth, T *values) : row_(rows), col_(cols), dep_(depth)
    {
        values_ = values;
    }

    template<typename T>
    T & matrix3<T>::operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error) {
#ifdef __CUDA_ARCH__
        if (row >= row_ || col >= col_ || dep >= dep_) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator()");
            return values_[0];
        }
        return values_[dep + col * dep_ + row * col_ * dep_];
#else
        if (row >= row_ || col >= col_ || dep >= dep_)
            throw MatrixError("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator() const");
        return values_[dep + col * dep_ + row * col_ * dep_];
#endif
    }

    template<typename T>
    const T & matrix3<T>::operator()(size_t row, size_t col, size_t dep, CudaError *cuda_error) const {
#ifdef __CUDA_ARCH__
        if (row >= row_ || col >= col_ || dep >= dep_) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator() const");
            return values_[0];
        }
        return values_[dep + col * dep_ + row * col_ * dep_];
#else
        if (row >= row_ || col >= col_ || dep >= dep_)
            throw MatrixError("Tried to access out of bounds element in matrix.", "Matrix3.cuh | matrix3::operator() const");
        return values_[dep + col * dep_ + row * col_ * dep_];
#endif
    }

}

#endif //MATRIX3_CUH
