/*
** RAYCASTING LIBRARY
** Triangle.hpp
** Created by marton on 20/05/24.
*/

#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "IMaterial.hpp"
#include "IShape.hpp"

namespace rcr {

    struct Triangle {
        vec3<float> *d_p1_ = nullptr;
        vec3<float> *d_p2_ = nullptr;
        vec3<float> *d_p3_ = nullptr;

        __host__ Triangle(vec3<float> p1, vec3<float> p2, vec3<float> p3);
        __host__ ~Triangle();

        __device__ hitPos hit(rcr::ray ray) const;
        __device__ float2 getUvMapping(vec3<float> pos) const;

        __host__ void changePosition(vec3<float> p1, vec3<float> p2, vec3<float> p3);
        __host__ void changeMaterial(IMaterial *mat);
    };

} // rcr

#endif //TRIANGLE_HPP
