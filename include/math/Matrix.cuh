/*
** RAYCASTING LIBRARY
** Matrix.cuh
** Created by marton on 15/06/24.
*/

#ifndef MATRIX_CUH
#define MATRIX_CUH

namespace rcr {

    template<u_int8_t ROW, u_int8_t COL, typename T>
    class Matrix {
            T d_values_[ROW * COL] = {};

    public:
        __device__ __host__ Matrix() = default;
        __device__ __host__ Matrix(T *values);

        __device__ __host__ T& operator()(u_int8_t row, u_int8_t col);
        __device__ __host__ const T& operator()(u_int8_t row, u_int8_t col) const;
    };

    template<u_int8_t ROW, u_int8_t COL, typename T>
    Matrix<ROW, COL, T>::Matrix(T *values) {
        for (int i = 0; i < ROW * COL; i++)
            d_values_[i] = values[i];
    }

    template<u_int8_t ROW, u_int8_t COL, typename T>
    T & Matrix<ROW, COL, T>::operator()(u_int8_t row, u_int8_t col) {
        return d_values_[col + row * COL];
    }

    template<u_int8_t ROW, u_int8_t COL, typename T>
    const T& Matrix<ROW, COL, T>::operator()(u_int8_t row, u_int8_t col) const {
        return d_values_[col + row * COL];
    }
}

#endif //MATRIX_CUH
