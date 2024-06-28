cmake_minimum_required(VERSION 3.27)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()
set(TCNN_CUDA_ARCHITECTURES 6.1)

set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(BUILD_SHARED_LIBS ON)

project(raycaster LANGUAGES CUDA CXX)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(raycaster
        src/main.cu
        include/CudaError.hpp
        include/math/Ray.cuh
        include/math/Vec3.cuh
        include/IMaterial.hpp
        src/render/render.cu
        src/shapes/Triangle.cu
        include/shapes/Triangle.hpp
        include/render/Renderer.cuh
        include/math/RGB.cuh
        include/math/Matrix2.cuh
        include/Error.hpp
        include/math/Matrix3.cuh
)

set_target_properties(raycaster PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(raycaster ${OpenCV_LIBS})
