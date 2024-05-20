/*
** RAYCASTING LIBRARY
** PhongMaterial.hpp
** Created by marton on 20/05/24.
*/

#ifndef RAYCASTER_PHONGMATERIAL_HPP
#define RAYCASTER_PHONGMATERIAL_HPP

#include "IMaterial.hpp"

namespace rcr {
    class PhongMaterial : public IMaterial {
    public:
        PhongMaterial() = default;

        __device__ ushort3 calcColor(
                int recursion,
                vec3<float> collision,
                vec3<float> normal,
                rcr::ray camera,
                rcr::IShape *shape,
                std::vector<IShape *> scene) override;
    };
}


#endif //RAYCASTER_PHONGMATERIAL_HPP
