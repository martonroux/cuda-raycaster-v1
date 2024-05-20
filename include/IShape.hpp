/*
** RAYCASTING LIBRARY
** IShape.hpp
** Created by marton on 20/05/24.
*/

#ifndef ISHAPE_HPP
#define ISHAPE_HPP

#include <math/Ray.hpp>
#include <vector>

namespace rcr {

    class IShape {
    public:
        virtual ~IShape() = default;

        /**
         * The hit() function checks whether a ray hits said shape, and if it does, returns the closest position.
         */
        __device__ virtual hitPos hit(ray ray) const = 0;

        /**
         * The getUvMapping() function returns the 2D UV coords depending on 3D coordinates, for the actual shape.
         */
        __device__ virtual float2 getUvMapping(vec3<float> pos) const = 0;
    };

}

#endif //ISHAPE_HPP
