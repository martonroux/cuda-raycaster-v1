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
         * Shapes can have multiple properties at a time. For this reason, we give multiple blocks to each shape
         * to calculate more efficiently. Each block should be assigned to one of the properties of the shape. If
         * there aren't enough blocks, the hit() method should account for it.
         */
        __device__ [[nodiscard]] hitPos hit(ray ray, int nbBlocks) const;

        /**
         * The amount() function returns the amount of different properties that the shape has. This will be used by
         * the calling function to determine how many blocks will be assigned for the hit() function of the shape.
         */
        __device__ [[nodiscard]] unsigned int amount() const;
    };

}

#endif //ISHAPE_HPP
