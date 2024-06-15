/*
** RAYCASTING LIBRARY
** Ray.cuh
** Created by marton on 20/05/24.
*/

#ifndef RAY_HPP
#define RAY_HPP

#include "Vec3.cuh"

namespace rcr {

    struct ray {
        vec3<float> origin;
        vec3<float> direction;
    };

    struct hitPos {
        bool hit;
        vec3<float> pos;
    };

}

#endif //RAY_HPP
