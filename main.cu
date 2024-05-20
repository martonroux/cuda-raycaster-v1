#include <iostream>
#include "CudaError.hpp"
#include "math/Ray.hpp"
#include "IShape.hpp"

__global__ void vec3test(rcr::vec3<float> *vec1, rcr::vec3<float> *vec2) {
    if (threadIdx.x == 0)
        printf("%f\n", vec1->length(threadIdx.x));
    rcr::vec3<float> ret = vec1->div(2.f, threadIdx.x);
    if (threadIdx.x == 0)
        printf("%f %f %f\n", ret.x, ret.y, ret.z);
}

int main() {
    rcr::vec3<float> h_vec1[] = {{1, 2, 3}};
    rcr::vec3<float> h_vec2[] = {{4, 5, 6}};

    rcr::vec3<float> *d_vec1; rcr::vec3<float> *d_vec2;

    cudaMalloc((void**)&d_vec1, sizeof(rcr::vec3<float>));
    cudaMalloc((void**)&d_vec2, sizeof(rcr::vec3<float>));

    cudaMemcpy(d_vec1, h_vec1, sizeof(rcr::vec3<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, sizeof(rcr::vec3<float>), cudaMemcpyHostToDevice);

    vec3test<<<1, 3>>>(d_vec1, d_vec2);
    cudaDeviceSynchronize();

    cudaFree(d_vec1); cudaFree(d_vec2);
    // free(h_vec1); free(h_vec2);
}

// int main() {
//     // Allocate CudaError object in host memory
//     rcr::CudaError* h_cudaError = new rcr::CudaError();
//
//     // Allocate CudaError object in device memory
//     rcr::CudaError* d_cudaError;
//     cudaMalloc((void**)&d_cudaError, sizeof(rcr::CudaError));
//
//     // Copy the host CudaError object to the device
//     cudaMemcpy(d_cudaError, h_cudaError, sizeof(rcr::CudaError), cudaMemcpyHostToDevice);
//
//     // Launch the kernel
//     test3<<<1, 1>>>(d_cudaError);
//     cudaDeviceSynchronize();
//
//     // Copy the CudaError object back to the host to check results
//     cudaMemcpy(h_cudaError, d_cudaError, sizeof(rcr::CudaError), cudaMemcpyDeviceToHost);
//
//     // Check if the exception was set correctly
//     try {
//         h_cudaError->throwException();
//     } catch (const rcr::CudaException& e) {
//         std::cout << "Caught CudaException: " << e.what() << " at " << e.where() << std::endl;
//     }
//
//     // Clean up
//     delete h_cudaError;
//     cudaFree(d_cudaError);
//
//     return 0;
// }

// int main() {
//     float h_mat1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//     float h_mat2[3][1] = {{1}, {2}, {3}};
//     float h_res[3][1] = {};
//     std::pair<int, int> mat1Size = {3, 3};
//     std::pair<int, int> mat2Size = {3, 1};
//
//     int numberBlocks = 1;
//     dim3 threadsPerBlock(mat1Size.second, mat2Size.first);
//
//     float *d_mat1, *d_mat2, *d_res;
//
//     cudaMalloc((void**)&d_mat1, mat1Size.first * mat1Size.second * sizeof(float));
//     cudaMalloc((void**)&d_mat2, mat2Size.first * mat2Size.second * sizeof(float));
//     cudaMalloc((void**)&d_res, mat1Size.second * mat2Size.first * sizeof(float));
//
//     cudaMemcpy(d_mat1, h_mat1, mat1Size.first * mat1Size.second * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_mat2, h_mat2, mat2Size.first * mat2Size.second * sizeof(float), cudaMemcpyHostToDevice);
//     rcr::matrix<3, 3, float> mat1 = {d_mat1};
//     rcr::matrix<3, 1, float> mat2 = {d_mat2};
//     rcr::matrix<3, 1, float> matRes = {d_res};
//     test<<<numberBlocks, threadsPerBlock>>>(mat1, mat2, matRes);
//
//     cudaDeviceSynchronize();
//
//     cudaMemcpy(h_res, d_res, mat1Size.second * mat2Size.first * sizeof(float), cudaMemcpyDeviceToHost);
//
//     for (int i = 0; i < mat1Size.first; i++) {
//         for (int j = 0; j < mat2Size.second; j++)
//             std::cout << h_res[i][j] << "\t";
//         std::cout << std::endl;
//     }
//
//     cudaFree(d_mat1);
//     cudaFree(d_mat2);
//     cudaFree(d_res);
//     return 0;
// }
