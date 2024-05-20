/*
** RAYCASTING LIBRARY
** IMaterial.hpp
** Created by marton on 20/05/24.
*/

#ifndef IMATERIAL_HPP
#define IMATERIAL_HPP

#include "IShape.hpp"
#include "math/Vec3.hpp"
#include "math/Ray.hpp"

namespace rcr {
    class IMaterial {
    public:
        virtual ~IMaterial() = default;

        /**
         * This is the main IMaterial function. It performs the most important calculation, which is the color of a
         * shape at a certain coordinate. It can be used recursively for reflections / refractions.
         */
        __device__ virtual ushort3 calcColor(
            int recursion,
            vec3<float> collision,
            vec3<float> normal,
            ray camera,
            IShape* shape,
            std::vector<IShape*> scene) = 0;
    };
}

#endif //IMATERIAL_HPP
