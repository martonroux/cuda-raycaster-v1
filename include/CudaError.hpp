/*
** RAYCASTING LIBRARY
** CudaError.hpp
** Created by marton on 20/05/24.
*/

#ifndef CUDAERROR_HPP
#define CUDAERROR_HPP

#define CUDA_ERROR_MSG_CAP_SIZE 256

#include <exception>
#include <cstring> // For std::memcpy
#include <string>

namespace rcr {

    class CudaException : public std::exception {
        char _what[CUDA_ERROR_MSG_CAP_SIZE] = {};
        char _where[CUDA_ERROR_MSG_CAP_SIZE] = {};

    public:
        CudaException(const char* what, const char* where) {
            std::memcpy(_what, what, sizeof(char) * CUDA_ERROR_MSG_CAP_SIZE);
            std::memcpy(_where, where, sizeof(char) * CUDA_ERROR_MSG_CAP_SIZE);
        }

        [[nodiscard]] const char * what() const noexcept override { return _what; }
        [[nodiscard]] const char * where() const noexcept { return _where; }
    };

    class CudaError {
        char _what[CUDA_ERROR_MSG_CAP_SIZE] = {};
        char _where[CUDA_ERROR_MSG_CAP_SIZE] = {};
        bool _hasError = false;

    public:
        CudaError() = default;

        __host__ static CudaError *createDeviceCudaError() {
            CudaError temp{};
            CudaError *d_error;

            cudaMalloc((void**)&d_error, sizeof(CudaError));
            cudaMemcpy(d_error, &temp, sizeof(CudaError), cudaMemcpyHostToDevice);
            return d_error;
        }

        __host__ static void checkDeviceCudaError(CudaError *d_error) {
            CudaError h_error{};

            cudaMemcpy(&h_error, d_error, sizeof(CudaError), cudaMemcpyDeviceToHost);

            try {
                h_error.throwException();
            } catch (const CudaException& e) {
                std::cerr << e.where() << ": " << e.what() << std::endl;
            }
        }

        __device__ __host__ bool hasError() const noexcept { return _hasError; }

        __device__ void setException(const char* what, const char* where) {
            if (_what[0] != '\0' || _where[0] != '\0')
                return;
            _hasError = true;

            for (int i = 0; i < CUDA_ERROR_MSG_CAP_SIZE; ++i) {
                _what[i] = what[i];
                _where[i] = where[i];
                if (what[i] == '\0' && where[i] == '\0') break;
            }
        }

        __host__ void throwException() const {
            if (_what[0] == '\0' && _where[0] == '\0')
                return;
            throw CudaException(_what, _where);
        }
    };

}

#endif // CUDAERROR_HPP
