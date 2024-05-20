/*
** RAYCASTING LIBRARY
** Vec3.hpp
** Created by marton on 20/05/24.
*/

#ifndef VEC3_HPP
#define VEC3_HPP

namespace rcr {

    template<typename T>
    struct vec3 {
        T x;
        T y;
        T z;

        __device__ float length(unsigned int threadIdx) {
            __shared__ float sum[3];

            if (threadIdx == 0)
                sum[0] = x * x;
            if (threadIdx == 1)
                sum[1] += y * y;
            if (threadIdx == 2)
                sum[2] += z * z;
            __syncthreads();
            return std::sqrt(sum[0] + sum[1] + sum[2]);
        }
        __device__ void normalize(unsigned int threadIdx) {
            float len = length(threadIdx);

            if (threadIdx == 0)
                x /= len;
            if (threadIdx == 1)
                y /= len;
            if (threadIdx == 2)
                z /= len;
        }
        __device__ float dot(const vec3* other, unsigned int threadIdx) {
            __shared__ vec3 ret;

            if (threadIdx == 0)
                ret.x = x * other->x;
            if (threadIdx == 1)
                ret.y = y * other->y;
            if (threadIdx == 2)
                ret.z = z * other->z;
            __syncthreads();
            return ret.x + ret.y + ret.z;
        }
        __device__ vec3 cross(const vec3* other, unsigned int threadIdx) {
            __shared__ vec3 ret;

            if (threadIdx == 0)
                ret.x = y * other->z - z * other->y;
            if (threadIdx == 1)
                ret.y = z * other->x - x * other->z;
            if (threadIdx == 2)
                ret.z = x * other->y - y * other->x;
            __syncthreads();
            return ret;
        }
        __device__ vec3 sum(const vec3* other, unsigned int threadIdx) {
            __shared__ vec3 ret;

            if (threadIdx == 0)
                ret.x = x + other->x;
            if (threadIdx == 1)
                ret.y = y + other->y;
            if (threadIdx == 2)
                ret.z = z + other->z;
            __syncthreads();
            return ret;
        }
        __device__ vec3 diff(const vec3* other, unsigned int threadIdx) {
            __shared__ vec3 ret;

            if (threadIdx == 0)
                ret.x = x - other->x;
            if (threadIdx == 1)
                ret.y = y - other->y;
            if (threadIdx == 2)
                ret.z = z - other->z;
            __syncthreads();
            return ret;
        }
        __device__ vec3 mult(T value, unsigned int threadIdx) {
            __shared__ vec3 ret;

            if (threadIdx == 0)
                ret.x = x * value;
            if (threadIdx == 1)
                ret.y = y * value;
            if (threadIdx == 2)
                ret.z = z * value;
            __syncthreads();
            return ret;
        }
        __device__ vec3 div(T value, unsigned int threadIdx) {
            __shared__ vec3 ret;

            if (threadIdx == 0)
                ret.x = x / value;
            if (threadIdx == 1)
                ret.y = y / value;
            if (threadIdx == 2)
                ret.z = z / value;
            __syncthreads();
            return ret;
        }
    };

}

#endif //VEC3_HPP
