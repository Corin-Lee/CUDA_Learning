#include <cuda_runtime.h>

__global__ void SoftmaxGpuV4(const float* in, float* out, const size_t n,
                             const size_t c) {
  extern __shared__ float shared[];  // 2倍线程束数量
  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  const int block_start = blockIdx.x * c;
  // 1. get local max value
  float local_max = -INFINITY;
  for (size_t i = tid; i < c; i += blockDim.x) {
    local_max = fmaxf(in[i + block_start], local_max);
  }

  // 2. get global max value
  for (int t = 16; t > 0; t /= 2) {
    local_max = fmaxf(local_max, __shfl_down_sync(~0, local_max, t));
  }

  if (lane_id == 0) {
    shared[warp_id] = local_max;
  }
  __syncthreads();

  float max_val = shared[warp_id];
  for (int i = 0; i < blockDim.x / 32; ++i) {
    max_val = fmaxf(max_val, shared[i]);
  }

  // 2. get sum & calculate
  float local_sum = 0.0f;
  for (size_t i = tid; i < c; i += blockDim.x) {
    out[i + block_start] = expf(in[i + block_start] - max_val);
    local_sum += out[i + block_start];
  }

  for (int t = 16; t > 0; t /= 2) {
    local_sum += __shfl_down_sync(~0, local_sum, t);
  }

  if (lane_id == 0) {
    shared[warp_id + blockDim.x / 32] = local_sum;
  }
  __syncthreads();

  float sum = 0.0f;
  for (size_t t = blockDim.x / 32; t < blockDim.x / 32 * 2; ++t) {
    sum += shared[t];
  }
  sum = 1.0f / sum;
  // calculate
  for (size_t i = tid; i < c; i += blockDim.x) {
    out[i + block_start] *= sum;
  }
}