#include <cuda_runtime.h>

__global__ void SoftmaxGpuV3(const float* in, float* out, const size_t n,
                             const size_t c) {
  const int block_start = blockIdx.x * c;
  const int tid = threadIdx.x;
  // 1. get local max value
  float local_max = -INFINITY;
  for (size_t i = tid; i < c; i += blockDim.x) {
    local_max = fmaxf(in[i + block_start], local_max);
  }

  // 2. get global max value
  size_t stride = blockDim.x;
  while (stride / 2 > 0) {
    stride /= 2;
    local_max = fmaxf(local_max, __shfl_down_sync(~0, local_max, stride));
  }
  float max_val = __shfl_sync(~0, local_max, 0);

  // 2. get sum & calculate
  float local_sum = 0.0f;
  stride = blockDim.x;
  for (size_t i = tid; i < c; i += stride) {
    out[i + block_start] = expf(in[i + block_start] - max_val);
    local_sum += out[i + block_start];
  }

  while (stride / 2 > 0) {
    stride /= 2;
    local_sum += __shfl_down_sync(~0, local_sum, stride);
  }

  float sum = 1.0f / __shfl_sync(~0, local_sum, 0);

  // calculate
  for (size_t i = tid; i < c; i += blockDim.x) {
    out[i + block_start] *= sum;
  }
}