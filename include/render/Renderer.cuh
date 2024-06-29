/*
** RAYCASTING LIBRARY
** Renderer.cuh
** Created by marton on 29/06/24.
*/

#ifndef RENDERER_CUH
#define RENDERER_CUH

#include "math/Matrix3.cuh"
#include "math/Ray.cuh"
#include "shapes/Triangle.hpp"
#include "render/rendererData.h"

namespace rcr {

    inline __device__ ray getPixelRay(float u, float v, rendererData data, CudaError *error)
    {
        if (u > 1.0 || u < 0.0 || v > 1.0 || v < 0.0) {
            error->setException("[INTERNAL] uv values are incorrect. Should be 0.0 >= u / v >= 1.0", "Renderer.cuh | getPixelRay");
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

    inline __device__ void render(matrix3<hitPos> *image, size_t height, size_t width, size_t numTriangles, Triangle *triangles,
        rendererData data, CudaError *error)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int triangleId = idx / (height * width);
        int pixels[2] = {};

        if (triangleId >= numTriangles) {
            error->setException("[INTERNAL] Tried to assign a triangle to a thread that doesn't exist.", "Renderer.cuh | render");
            return;
        }
        idx = idx % (height * width);

        pixels[0] = idx % width;
        pixels[1] = idx / width;

        if (pixels[0] >= width || pixels[1] >= height) {
            error->setException("[INTERNAL] Tried to assign a pixel to a thread that doesn't exist.", "Renderer.cuh | render");
            return;
        }
        float u = pixels[0] / static_cast<float>(width - 1);
        float v = pixels[1] / static_cast<float>(height - 1);

        ray ray = getPixelRay(u, v, data, error);
        hitPos pos = triangles[triangleId].hit(ray);

        (*image)(pixels[1], pixels[0], triangleId, error).hit = pos.hit;
        (*image)(pixels[1], pixels[0], triangleId, error).pos = pos.pos;
    }

    inline __global__ void kernelRender(matrix3<rcr::hitPos> *image, size_t height, size_t width, size_t numTriangles,
        rcr::Triangle *triangles, rcr::rendererData screen, rcr::CudaError *error)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx == 0) {
            printf("%f %f %f\n", triangles[0].getP1().x, triangles[0].getP1().y, triangles[0].getP1().z);
        }
        if (idx >= numTriangles * height * width)
            return;
        render(image, height, width, numTriangles, triangles, screen, error);
    }

}

#endif //RENDERER_CUH
