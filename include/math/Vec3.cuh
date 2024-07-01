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

        [[nodiscard]] __device__ __host__ float length() const {
            vec3<float> len{};
            len.x = x * x;
            len.y = y * y;
            len.z = z * z;
            return std::sqrt(len.x + len.y + len.z);
        }

        __device__ __host__ void normalize() const {
            float len;

            len = length();
            x /= len;
            y /= len;
            z /= len;
        }

        [[nodiscard]] __device__ __host__ float dot(const vec3* other) const {
            float res = 0;

            res += x * other->x;
            res += y * other->y;
            res += z * other->z;

            return res;
        }

        __device__ __host__ void cross(const vec3* other, vec3* res) const {
            res->x = y * other->z - z * other->y;
            res->y = z * other->x - x * other->z;
            res->z = x * other->y - y * other->x;
        }

        __device__ __host__ void sum(const vec3* other, vec3* res) const {
            res->x = x + other->x;
            res->y = y + other->y;
            res->z = z + other->z;
        }

        __device__ __host__ void diff(const vec3* other, vec3* res) const {
            res->x = x - other->x;
            res->y = y - other->y;
            res->z = z - other->z;
        }

        __device__ __host__ void mult(T value, vec3* res) const {
            res->x = x * value;
            res->y = y * value;
            res->z = z * value;
        }

        __device__ __host__ void div(T value, vec3* res) const {
            res->x = x / value;
            res->y = y / value;
            res->z = z / value;
        }

        [[nodiscard]] __device__ __host__ float getDistance(vec3 *other) const {
            vec3 temp{};

            this->diff(other, &temp);
            return temp.length();
        }
    };

}

#endif //VEC3_HPP
