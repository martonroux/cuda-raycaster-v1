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
            matrixh<H, W, hitPos> *image,
            Triangle *triangles,
            unsigned int nbTriangles,
            rendererData data,
            CudaError *error) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int nbElems = static_cast<int>(nbTriangles);
        int triangleId = idx % nbElems;
        int pixels[2] = {};

        pixels[0] = idx % W;
        pixels[1] = idx / W;

        float u = pixels[0] / static_cast<float>(W - 1);
        float v = pixels[1] / static_cast<float>(H - 1);

        ray ray = getPixelRay(u, v, data, error);
        hitPos pos = triangles[triangleId].hit(ray);

        (*image)(pixels[0], pixels[1]).hit = pos.hit;
        (*image)(pixels[0], pixels[1]).pos = pos.pos;
    }

} // rcr

#endif //RENDERER_HPP
