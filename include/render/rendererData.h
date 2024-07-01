/*
** RAYCASTING LIBRARY
** rendererData.h
** Created by marton on 29/06/24.
*/

#ifndef SCREENDATA_H
#define SCREENDATA_H

#include "math/Vec3.cuh"
#include "math/RGB.cuh"

namespace rcr {

    typedef struct {
        vec3<float> topLeft;
        vec3<float> width;
        vec3<float> height;
    } screenData;

    typedef struct {
        vec3<float> camPos;
        rgb backgroundColor;
        screenData screen;
    } rendererData;

}

#endif //SCREENDATA_H
