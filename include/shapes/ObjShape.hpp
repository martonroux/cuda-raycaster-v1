/*
** RAYCASTING LIBRARY
** ObjShape.hpp
** Created by marton on 01/07/24.
*/

#ifndef OBJSHAPE_HPP
#define OBJSHAPE_HPP

#include "shapes/Triangle.hpp"
#include "math/Matrix2.cuh"
#include <vector>

namespace rcr {

    class ObjShape {
        std::vector<Triangle> triangles_;

    public:
        ObjShape() = default;
        ~ObjShape() = default;

        void addTriangle(Triangle triangle);
        void applyTransformation(matrix2<float> transformation);

        [[nodiscard]] std::vector<Triangle> getTriangles() const { return triangles_; }
    };

} // rcr

#endif //OBJSHAPE_HPP
