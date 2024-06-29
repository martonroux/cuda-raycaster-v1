/*
** RAYCASTING LIBRARY
** rendererData.h
** Created by marton on 29/06/24.
*/

#ifndef SCREENDATA_H
#define SCREENDATA_H

#include "math/Vec3.cuh"

namespace rcr {

    typedef struct {
        rcr::vec3<float> topLeft;
        rcr::vec3<float> width;
        rcr::vec3<float> height;
    } screenData;

    typedef struct {
        rcr::vec3<float> camPos;
        screenData screen;
    } rendererData;

}

#endif //SCREENDATA_H
