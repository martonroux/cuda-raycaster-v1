/*
** RAYCASTING LIBRARY
** Triangle.hpp
** Created by marton on 15/06/24.
*/

#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "math/Ray.hpp"
#include "math/Vec3.hpp"

namespace rcr {

    class Triangle {
        vec3<float> p1_;
        vec3<float> p2_;
        vec3<float> p3_;

    public:
        __host__ Triangle(vec3<float> p1, vec3<float> p2, vec3<float> p3) : p1_(p1), p2_(p2), p3_(p3) {}
        __host__ ~Triangle() = default;

        __device__ hitPos hit(ray ray);
    };

} // rcr

#endif //TRIANGLE_HPP
