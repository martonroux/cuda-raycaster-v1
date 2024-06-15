/*
** RAYCASTING LIBRARY
** IMaterial.hpp
** Created by marton on 20/05/24.
*/

#ifndef IMATERIAL_HPP
#define IMATERIAL_HPP

#include "math/Vec3.hpp"
#include "math/Ray.hpp"

/**
 * As you can see, this is not a classic interface like we're used to. This is because CUDA has very minimal
 * support (or no support at all) for virtual functions that are on __device__. We could technically
 * create an interface with the virtual functions being only the ones run on __host__, but that wouldn't be very
 * useful (PS: __host__ is the CPU, __device__ the GPU).
 *
 * Here is a workaround that I found, and although it is pretty messy, it works and allows to add multiple
 * types of Materials without having to declare them manually in the code (shared libraries will work).
 * Same goes for Shapes.
 */

// namespace rcr {
//
//     /**
//      * This is the main IMaterial function. It performs the most important calculation, which is the color of a
//      * shape at a certain coordinate. It can be used recursively for reflections / refractions.
//      */
//     template<typename Material, typename Shape>
//     __device__ ushort3 calcColor(
//         const Material* mat,
//         int recursion,
//         vec3<float> collision,
//         vec3<float> normal,
//         ray camera,
//         Shape* shape,
//         std::vector<Shape*> scene) {
//         return mat->calcColor(recursion, collision, normal, camera, shape, scene);
//     }
//
// }

#endif //IMATERIAL_HPP
