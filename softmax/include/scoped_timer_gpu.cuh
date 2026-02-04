#pragma once
#include <cuda_runtime.h>

#include <string>

class ScopedTimerGpu {
 public:
  explicit ScopedTimerGpu(const std::string& name) : name_(name) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~ScopedTimerGpu() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_, stop_);
    printf("[%s] GPU cost: %.6f ms\n", name_.c_str(), milliseconds);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

 private:
  std::string name_;
  cudaEvent_t start_, stop_;
};