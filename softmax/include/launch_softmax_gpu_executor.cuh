#pragma once
#include <cuda_runtime.h>

#include <string>

#include "scoped_timer_gpu.cuh"

struct SoftmaxGpuKernelConfig {
  dim3 grid = 0;
  dim3 block = 0;
  size_t shared_mem = 0;
};

template <typename Func>
class LaunchSoftmaxGpuExecutor {
 public:
  LaunchSoftmaxGpuExecutor(Func func, const std::string& func_name,
                           const float* in, float* out, const size_t n,
                           const size_t c)
      : func_(func), func_name_(func_name), in_(in), out_(out), n_(n), c_(c) {}

  void run(SoftmaxGpuKernelConfig config) {
    float* d_out;
    const size_t bytes_nums = n_ * c_ * sizeof(float);
    cudaMalloc(&d_out, bytes_nums);
    {
      ScopedTimerGpu gpu_timer(func_name_);
      func_<<<config.grid, config.block, config.shared_mem>>>(in_, d_out, n_,
                                                              c_);
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
      printf("[%s] runs failed!\n", func_name_.c_str());
    }
    cudaMemcpy(out_, d_out, bytes_nums, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
  }

 private:
  Func func_;
  std::string func_name_ = "";
  const float* in_;
  float* out_;
  const size_t n_;
  const size_t c_;
};