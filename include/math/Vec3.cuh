/*
** RAYCASTING LIBRARY
** Vec3.cuh
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

        // THIS FUNCTION IS BLOCKING
        __device__ float length(unsigned int threadIdx, unsigned int startThreadIdx) {
            __shared__ vec3<float> len;

            len = {};
            __syncthreads();
            if (threadIdx == startThreadIdx)
                len.x = x * x;
            else if (threadIdx == startThreadIdx + 1)
                len.y = y * y;
            else if (threadIdx == startThreadIdx + 2)
                len.z = z * z;
            __syncthreads();
            return std::sqrt(len.x + len.y + len.z);
        }

        // THIS FUNCTION IS BLOCKING
        __device__ void normalize(unsigned int threadIdx, unsigned int startThreadIdx) {
            __shared__ float len;

            len = length(threadIdx, startThreadIdx);
            __syncthreads();
            if (threadIdx == startThreadIdx)
                x /= len;
            else if (threadIdx == startThreadIdx + 1)
                y /= len;
            else if (threadIdx == startThreadIdx + 2)
                z /= len;
        }

        // THIS FUNCTION IS BLOCKING
        __device__ float dot(const vec3* other, unsigned int threadIdx, unsigned int startThreadIdx) {
            __shared__ float res;

            res = 0;
            __syncthreads();

            if (threadIdx == startThreadIdx) {
                res += x * other->x;
            }
            else if (threadIdx == startThreadIdx + 1)
                res += y * other->y;
            else if (threadIdx == startThreadIdx + 2)
                res += z * other->z;

            __syncthreads();
            return res;
        }

        __device__ float dotSingle(const vec3* other) {
            float res = 0;

            res += x * other->x;
            res += y * other->y;
            res += z * other->z;

            return res;
        }
        __device__ void cross(const vec3* other, vec3* res, unsigned int threadIdx, unsigned int startThreadIdx) {
            if (threadIdx == startThreadIdx)
                res->x = y * other->z - z * other->y;
            else if (threadIdx == startThreadIdx + 1)
                res->y = z * other->x - x * other->z;
            else if (threadIdx == startThreadIdx + 2)
                res->z = x * other->y - y * other->x;
        }
        __device__ void crossSingle(const vec3* other, vec3* res) {
            res->x = y * other->z - z * other->y;
            res->y = z * other->x - x * other->z;
            res->z = x * other->y - y * other->x;
        }
        __device__ void sum(const vec3* other, vec3* res, unsigned int threadIdx, unsigned int startThreadIdx) {
            if (threadIdx == startThreadIdx)
                res->x = x + other->x;
            else if (threadIdx == startThreadIdx + 1)
                res->y = y + other->y;
            else if (threadIdx == startThreadIdx + 2)
                res->z = z + other->z;
        }
        __device__ void sumSingle(const vec3* other, vec3* res) {
            res->x = x + other->x;
            res->y = y + other->y;
            res->z = z + other->z;
        }
        __device__ void diff(const vec3* other, vec3* res, unsigned int threadIdx, unsigned int startThreadIdx) {
            if (threadIdx == startThreadIdx)
                res->x = x - other->x;
            else if (threadIdx == startThreadIdx + 1)
                res->y = y - other->y;
            else if (threadIdx == startThreadIdx + 2)
                res->z = z - other->z;
        }
        __device__ void diffSingle(const vec3* other, vec3* res) {
            res->x = x - other->x;
            res->y = y - other->y;
            res->z = z - other->z;
        }
        __device__ void mult(T value, vec3* res, unsigned int threadIdx, unsigned int startThreadIdx) {
            if (threadIdx == startThreadIdx)
                res->x = x * value;
            else if (threadIdx == startThreadIdx + 1)
                res->y = y * value;
            else if (threadIdx == startThreadIdx + 2)
                res->z = z * value;
        }
        __device__ void multSingle(T value, vec3* res) {
            res->x = x * value;
            res->y = y * value;
            res->z = z * value;
        }
        __device__ void div(T value, vec3* res, unsigned int threadIdx, unsigned int startThreadIdx) {
            if (threadIdx == startThreadIdx)
                res->x = x / value;
            else if (threadIdx == startThreadIdx + 1)
                res->y = y / value;
            else if (threadIdx == startThreadIdx + 2)
                res->z = z / value;
        }
    };

}

#endif //VEC3_HPP
