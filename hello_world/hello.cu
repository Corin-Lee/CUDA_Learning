#include <cuda_runtime.h>

#include <iostream>

__global__ void HelloCuda() {
    int idx = threadIdx.x;
    printf("Hello World From GPU thread%d.\n", idx);

}

int main() {
    std::cout << "Hello World From CPU." << std::endl;
    std::cout << "=============================" << std::endl;

    HelloCuda<<<1, 10>>>();
    cudaDeviceSynchronize();

    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "CUDA err:  " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return 1;
    }
    std::cout << "=============================" << std::endl;
    std::cout << "GPU: Hello CUDA finished!" << std::endl;
    std::cout << "CPU: Hello CUDA finished!" << std::endl;

    return 0;
}