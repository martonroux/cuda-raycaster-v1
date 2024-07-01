/*
** RAYCASTING LIBRARY
** Matrix2.cuh
** Created by marton on 15/06/24.
*/

#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "Error.hpp"
#include "CudaError.hpp"

namespace rcr {

    /* -------------------------- 2D MATRIX -------------------------- */
    template<typename T>
    class matrix2 {
        size_t row_;
        size_t col_;

        T *values_ = nullptr;

    public:
        __host__ __device__ matrix2(size_t rows, size_t cols);
        __host__ __device__ matrix2(size_t rows, size_t cols, T *values);
        __host__ __device__ ~matrix2() = default;

        __host__ __device__ [[nodiscard]] T *getValues() const { return values_; }
        __host__ __device__ [[nodiscard]] size_t getNumRows() const { return row_; }
        __host__ __device__ [[nodiscard]] size_t getNumCols() const { return col_; }

        __host__ __device__ T& operator()(size_t row, size_t col, CudaError *cuda_error);
        __host__ __device__ const T& operator()(size_t row, size_t col, CudaError *cuda_error) const;

        __host__ __device__ void mult(matrix2 *other, matrix2 *res, CudaError *cuda_error);
    };

    template<typename T>
    __host__ __device__ ::rcr::matrix2<T>::matrix2(size_t rows, size_t cols) : row_(rows), col_(cols)
    {
        values_ = (T*)malloc(sizeof(T) * row_ * col_);

#ifndef __CUDA_ARCH__
        if (values_ == nullptr)
            throw MatrixError("Host memory allocation failed", "matrix2.cuh | matrix2::matrix2");
#endif
    }

    template<typename T>
    __host__ __device__ ::rcr::matrix2<T>::matrix2(size_t rows, size_t cols, T *values) : row_(rows), col_(cols)
    {
        values_ = values;
    }

    template<typename T>
    T & matrix2<T>::operator()(size_t row, size_t col, CudaError *cuda_error) {
#ifdef __CUDA_ARCH__
        if (row >= row_ || col >= col_) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "matrix2.cuh | matrix2::operator()");
            return values_[0];
        }
        return values_[col + row * col_];
#else
        if (row >= row_ || col >= col_)
            throw MatrixError("Tried to access out of bounds element in matrix.", "matrix2.cuh | matrix2::operator() const");
        return values_[col + row * col_];
#endif
    }

    template<typename T>
    const T & matrix2<T>::operator()(size_t row, size_t col, CudaError *cuda_error) const {
#ifdef __CUDA_ARCH__
        if (row >= row_ || col >= col_) {
            cuda_error->setException("Tried to access out of bounds element in matrix.", "matrix2.cuh | matrix2::operator() const");
            return values_[0];
        }
        return values_[col + row * col_];
#else
        if (row >= row_ || col >= col_)
            throw MatrixError("Tried to access out of bounds element in matrix.", "matrix2.cuh | matrix2::operator() const");
        return values_[col + row * col_];
#endif
    }

    template<typename T>
    void matrix2<T>::mult(matrix2 *other, matrix2 *res, CudaError *cuda_error) {
        if (other->getNumRows() != col_) {
#ifdef __CUDA_ARCH__
            cuda_error->setException("Tried to multiply matrixes that have incompatible sizes.", "Matrix2.cuh | matrix2::mult()");
            return;
#else
            throw MatrixError("Tried to multiply matrixes that have incompatible sizes.", "Matrix2.cuh | matrix2::mult()");
#endif
        } if (res->getNumRows() != row_ || res->getNumCols() != other->getNumCols()) {
#ifdef __CUDA_ARCH__
            cuda_error->setException("Result matrix of multiplication has invalid size.", "Matrix2.cuh | matrix2::mult()");
            return;
#else
            throw MatrixError("Result matrix of multiplication has invalid size.", "Matrix2.cuh | matrix2::mult()");
#endif
        }

        for (int i = 0; i < row_; i++) {
            for (int j = 0; j < other->getNumCols(); j++) {
                (*res)(i, j, cuda_error) = 0;

                for (int x = 0; x < col_; x++) {
                    (*res)(i, j, cuda_error) += (*this)(i, x, cuda_error) * (*other)(x, j, cuda_error);
                }
            }
        }
    }
}

#endif //MATRIX_CUH
