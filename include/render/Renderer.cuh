/*
** RAYCASTING LIBRARY
** Renderer.cuh
** Created by marton on 15/06/24.
*/

#ifndef RENDERER_HPP
#define RENDERER_HPP

#include "CudaError.hpp"
#include "shapes/Triangle.hpp"
#include "shapes/Triangle.hpp"
#include "math/Matrix.cuh"
#include "math/RGB.cuh"

namespace rcr {

    typedef struct {
        vec3<float> topLeft;
        vec3<float> width;
        vec3<float> height;
    } screenData;

    typedef struct {
        vec3<float> camPos;
        screenData screen;
    } rendererData;

    inline __device__ ray getPixelRay(float u, float v, rendererData data, CudaError *error) {
        if (u > 1.0 || u < 0.0 || v > 1.0 || v < 0.0) {
            error->setException("[INTERNAL] uv values are incorrect. Should be 0.0 >= u / v >= 1.0", "Renderer | getPixelRay");
            return {};
        }

        vec3<float> widthPart={}, heightPart={}, combination={}, result={}, finalDirection={};

        data.screen.width.multSingle(u, &widthPart);
        data.screen.height.multSingle(v, &heightPart);

        widthPart.sumSingle(&heightPart, &combination);
        combination.sumSingle(&data.screen.topLeft, &result);
        result.diffSingle(&data.camPos, &finalDirection);

        return {data.camPos, finalDirection};
    }

    template<size_t H, size_t W>
    __device__ void render(
            matrix<H, W, hitPos> *image,
            Triangle *triangles,
            unsigned int nbTriangles,
            rendererData data,
            CudaError *error) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int nbElems = static_cast<int>(nbTriangles);
        int triangleId = idx % nbElems;
        matrix<1, 2, int> pixel{};

        pixel(0, 0) = idx % W;
        pixel(0, 1) = idx / W;

        float u = pixel(0, 0) / static_cast<float>(W - 1);
        float v = pixel(0, 1) / static_cast<float>(H - 1);

        ray ray = getPixelRay(u, v, data, error);
        hitPos pos = triangles[triangleId].hit(ray);

        // printf("Thread %d at pixel (%d,%d), ray: (%f,%f,%f), hit: %d\n", idx, idx % static_cast<int>(W), idx / static_cast<int>(W), ray.direction.x, ray.direction.y, ray.direction.z, pos.hit);

        (*image)(pixel(0, 0), pixel(0, 1)).hit = pos.hit;
        (*image)(pixel(0, 0), pixel(0, 1)).pos = pos.pos;
    }

} // rcr

#endif //RENDERER_HPP
