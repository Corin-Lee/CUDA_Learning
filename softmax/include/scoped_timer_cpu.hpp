#pragma once

#include <chrono>
#include <string>

class ScopedTimerCpu {
 public:
  explicit ScopedTimerCpu(const std::string& name)
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

  ~ScopedTimerCpu() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cost = end - start_;
    printf("[%s] CPU cost: %.6f ms\n", name_.c_str(), cost.count());
  }

 private:
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
