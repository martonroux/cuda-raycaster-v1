cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()
set(TCNN_CUDA_ARCHITECTURES 6.1)
project(raycaster LANGUAGES CUDA CXX)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(raycaster src/main.cu
        include/CudaError.hpp
        include/math/Ray.hpp
        include/math/Vec3.hpp
        include/IShape.hpp
        include/shapes/Triangle.hpp
        include/IMaterial.hpp
        src/shapes/Triangle.cu
        src/materials/PhongMaterial.cu
        include/materials/PhongMaterial.hpp
        src/render/render.cu
)

set_target_properties(raycaster PROPERTIES CUDA_SEPARABLE_COMPILATION ON)