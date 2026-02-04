#include "scoped_timer_gpu.cuh"
#include "softmax_gpu.cuh"

void LaunchSoftmaxGpuV1(const float* in, float* out, const size_t n,
                        const size_t c) {
  float* d_out;
  const size_t bytes_nums = n * c * sizeof(float);
  cudaMalloc(&d_out, bytes_nums);
  {
    ScopedTimerGpu("SoftmaxGpuV1");
    SoftmaxGpuV1<<<n, 1>>>(in, d_out, n, c);
  }
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) {
    printf("[SoftmaxGpuV1] runs failed!\n");
  }
  cudaMemcpy(out, d_out, bytes_nums, cudaMemcpyDeviceToHost);
  cudaFree(d_out);
}

void LaunchSoftmaxGpuV2(const float* in, float* out, const size_t n,
                        const size_t c) {
  float* d_out;
  const size_t bytes_nums = n * c * sizeof(float);
  cudaMalloc(&d_out, bytes_nums);
  {
    ScopedTimerGpu("SoftmaxGpuV2");
    SoftmaxGpuV2<<<n, 32, 32 * sizeof(float)>>>(in, d_out, n, c);
  }
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) {
    printf("[SoftmaxGpuV2] runs failed!\n");
  }
  cudaMemcpy(out, d_out, bytes_nums, cudaMemcpyDeviceToHost);
  cudaFree(d_out);
}