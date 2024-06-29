/*
** RAYCASTING LIBRARY
** Triangle.hpp
** Created by marton on 15/06/24.
*/

#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "math/Ray.cuh"
#include "math/Vec3.cuh"

namespace rcr {

    class Triangle {
        vec3<float> p1_;
        vec3<float> p2_;
        vec3<float> p3_;

        unsigned int id_ = 0;

    public:
        __host__ Triangle(vec3<float> p1, vec3<float> p2, vec3<float> p3) : p1_(p1), p2_(p2), p3_(p3) {}
        __host__ ~Triangle() = default;

        __device__ hitPos hit(ray ray) const;
        __device__ vec3<float> getP1() const { return p1_; }
    };

} // rcr

#endif //TRIANGLE_HPP
