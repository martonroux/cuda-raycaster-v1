/*
** RAYCASTING LIBRARY
** Triangle.hpp
** Created by marton on 15/06/24.
*/

#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "math/Ray.cuh"
#include "math/Vec3.cuh"
#include "math/RGB.cuh"

namespace rcr {

    class Triangle {
        vec3<float> p1_;
        vec3<float> p2_;
        vec3<float> p3_;
        rgb color_;

        unsigned int id_ = 0;

    public:
        __host__ Triangle(vec3<float> p1, vec3<float> p2, vec3<float> p3, rgb color) : p1_(p1), p2_(p2), p3_(p3), color_(color) {}
        __host__ ~Triangle() = default;

        __device__ hitPos hit(ray ray) const;
        __device__ vec3<float> getP1() const { return p1_; }

        __device__ rgb getColor() const { return color_; }
    };

} // rcr

#endif //TRIANGLE_HPP
