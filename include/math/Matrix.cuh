/*
** RAYCASTING LIBRARY
** Matrix.cuh
** Created by marton on 15/06/24.
*/

#ifndef MATRIX2_CUH
#define MATRIX2_CUH

namespace rcr {

    template<size_t ROW, size_t COL, typename T>
    class matrix {
        T h_values_[ROW * COL] = {};
        T d_values_[ROW * COL] = {};

    public:
        __device__ __host__ matrix() = default;
        __device__ __host__ matrix(T *values);

        __host__ void moveToDevice();
        __host__ void moveToHost();

        __device__ __host__ T& operator()(size_t row, size_t col);
        __device__ __host__ const T& operator()(size_t row, size_t col) const;
    };

    template<size_t ROW, size_t COL, typename T>
    matrix<ROW, COL, T>::matrix(T *values) {
#ifdef __CUDACC__
        for (int i = 0; i < ROW * COL; i++)
            d_values_[i] = values[i];
#else
        for (int i = 0; i < ROW * COL; i++)
            h_values_[i] = values[i];
#endif
    }

    template<size_t ROW, size_t COL, typename T>
    void matrix<ROW, COL, T>::moveToDevice() {
        cudaMalloc((void**)&d_values_, sizeof(T) * ROW * COL);
        cudaMemcpy(d_values_, h_values_, sizeof(T) * ROW * COL, cudaMemcpyHostToDevice);
    }

    template<size_t ROW, size_t COL, typename T>
    void matrix<ROW, COL, T>::moveToHost() {
        cudaMemcpy(h_values_, d_values_, sizeof(T) * ROW * COL, cudaMemcpyDeviceToHost);
    }

    template<size_t ROW, size_t COL, typename T>
    T & matrix<ROW, COL, T>::operator()(size_t row, size_t col) {
#ifdef __CUDACC__
        return d_values_[col + row * COL];
#else
        return h_values_[col + row * COL];
#endif
    }

    template<size_t ROW, size_t COL, typename T>
    const T& matrix<ROW, COL, T>::operator()(size_t row, size_t col) const {
#ifdef __CUDACC__
        return d_values_[col + row * COL];
#else
        return h_values_[col + row * COL];
#endif
    }
}

#endif //MATRIX2_CUH
