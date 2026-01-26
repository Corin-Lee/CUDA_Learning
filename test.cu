#include <cuda_runtime.h>

#include <iostream>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    std::cout << "Device " << i << ": " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << "  Total SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Shared Memory per SM: "
              << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
    std::cout << "  Max Shared Memory per Block: "
              << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "  Max Registers per Block: " << prop.regsPerBlock / 1024
              << " K" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock
              << std::endl;
    std::cout << "  L2 Cache Size: " << prop.l2CacheSize / (1024.0 * 1024.0)
              << " MB" << std::endl;
  }
  return 0;
}