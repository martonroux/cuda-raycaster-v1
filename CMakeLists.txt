cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)

# Set CUDA architectures explicitly
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()
set(TCNN_CUDA_ARCHITECTURES 6.1)
project(cuda_test LANGUAGES CUDA)

# Specify the CUDA compiler explicitly

# Add an executable
add_executable(cuda_test main.cu)

# Enable CUDA for the target
set_target_properties(cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
