/*
** RAYCASTING LIBRARY
** ObjShape.cpp
** Created by marton on 01/07/24.
*/

#include "shapes/ObjShape.hpp"

namespace rcr {
    void ObjShape::addTriangle(Triangle triangle) {
        triangles_.push_back(triangle);
    }

    void ObjShape::applyTransformation(matrix2<float> transformation) {
        // TODO
    }
} // rcr
