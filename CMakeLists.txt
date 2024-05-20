cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()
set(TCNN_CUDA_ARCHITECTURES 6.1)
project(raycaster LANGUAGES CUDA CXX)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(raycaster main.cu
        old_include/math/Matrix.hpp
        old_include/math/Vector.hpp
        old_include/CudaVariable.hpp
        include/CudaError.hpp
        old_include/IShape.hpp
        include/math/Ray.hpp
        include/math/Vec3.hpp
        include/IShape.hpp
        src/shapes/Triangle.cpp
        include/shapes/Triangle.hpp
        include/IMaterial.hpp
)

set_target_properties(raycaster PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
