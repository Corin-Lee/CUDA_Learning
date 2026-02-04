#include <cuda_runtime.h>

#include <cstddef>

#include "softmax_gpu.cuh"

// n * c, 假设每行初始化 32 个线程, 每行数据量是32的倍数
__global__ void SoftmaxGpuV2(const float* in, float* out, const size_t n,
                             const size_t c) {
  extern __shared__ float cur_row[];
  const int block_start = blockIdx.x * c;
  const int tid = threadIdx.x;
  // 1. get max value
  float local_max = -INFINITY;
  for (size_t i = tid; i < c; i += blockDim.x) {
    local_max = fmaxf(in[i + block_start], local_max);
  }
  cur_row[tid] = local_max;
  __syncthreads();

  // Block 内规约 Max
  size_t stride = blockDim.x;
  while (stride / 2 > 0) {
    stride /= 2;
    if (tid < stride) {
      cur_row[tid] = fmaxf(cur_row[tid], cur_row[tid + stride]);
    }
    __syncthreads();
  }
  float max_val = cur_row[0];
  __syncthreads();

  // 2. get sum & calculate
  float local_sum = 0.0f;
  stride = blockDim.x;
  // 写入共享内存的初始值
  for (size_t i = tid; i < c; i += stride) {
    out[i + block_start] = expf(in[i + block_start] - max_val);
    local_sum += out[i + block_start];
  }
  cur_row[tid] = local_sum;
  __syncthreads();

  while (stride / 2 > 0) {
    stride /= 2;
    if (tid < stride) {
      cur_row[tid] += cur_row[tid + stride];
    }
    __syncthreads();
  }
  float sum = 1.0f / cur_row[0];

  // calculate
  for (size_t i = tid; i < c; i += blockDim.x) {
    out[i + block_start] *= sum;
  }
}