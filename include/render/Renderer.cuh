/*
** RAYCASTING LIBRARY
** Renderer.cuh
** Created by marton on 29/06/24.
*/

#ifndef RENDERER_CUH
#define RENDERER_CUH

#include "math/Matrix2.cuh"
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

        data.screen.width.mult(u, &widthPart);
        data.screen.height.mult(v, &heightPart);

        widthPart.sum(&heightPart, &combination);
        combination.sum(&data.screen.topLeft, &result);
        result.diff(&data.camPos, &finalDirection);

        return {data.camPos, finalDirection};
    }

    inline __device__ void hitDetect(matrix3<hitPos> *hits, size_t height, size_t width, size_t numTriangles, Triangle *triangles,
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

        (*hits)(pixels[1], pixels[0], triangleId, error).hit = pos.hit;
        (*hits)(pixels[1], pixels[0], triangleId, error).pos = pos.pos;
    }

    inline __device__ void render(matrix2<rgb> *image, matrix3<hitPos> *hits, size_t height, size_t width, size_t numTriangles,
        Triangle *triangles, rendererData screen, CudaError *error)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int pixels[2] = {};

        pixels[0] = idx % width;
        pixels[1] = idx / width;

        if (pixels[1] > height)
            return;

        int smallestIdx = -1;
        float smallestDistance = 0;
        float distance;

        for (int i = 0; i < numTriangles; i++) {
            if ((*hits)(pixels[1], pixels[0], i, error).hit == true) {
                if (smallestIdx == -1) {
                    smallestIdx = i;
                    smallestDistance = (*hits)(pixels[1], pixels[0], i, error).pos.getDistance(&screen.camPos);
                } else {
                    distance = (*hits)(pixels[1], pixels[0], i, error).pos.getDistance(&screen.camPos);

                    if (distance < smallestDistance) {
                        smallestIdx = i;
                        smallestDistance = distance;
                    }
                }
            }
        }

        if (smallestIdx != -1) {
            (*image)(pixels[1], pixels[0], error) = triangles[smallestIdx].getColor();
        } else {
            (*image)(pixels[1], pixels[0], error) = screen.backgroundColor;
        }
    }

    inline __global__ void kernelHitdetect(hitPos *hits, size_t height, size_t width, size_t numTriangles,
        Triangle *triangles, rendererData screen, CudaError *error)
    {
        matrix3 hitsMat{height, width, numTriangles, hits};
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= numTriangles * height * width)
            return;
        hitDetect(&hitsMat, height, width, numTriangles, triangles, screen, error);
    }

    inline  __global__ void kernelRender(rgb *image, hitPos *hits, size_t height, size_t width, size_t numTriangles,
        Triangle *triangles, rendererData screen, CudaError *error)
    {
        matrix3 hitsMat{height, width, numTriangles, hits};
        matrix2 imageMat{height, width, image};
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= height * width)
            return;
        render(&imageMat, &hitsMat, height, width, numTriangles, triangles, screen, error);
    }

}

#endif //RENDERER_CUH
