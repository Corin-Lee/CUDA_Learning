#include <cuda_runtime.h>

__global__ void SoftmaxGpuV3(const float* in, float* out, const size_t n,
                             const size_t c) {}