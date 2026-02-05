#pragma once
#include <iostream>
#include <memory>
#include <string>

#include "test.hpp"

template <typename Func>
void ExecuteSoftmaxTest(const std::string& func_name, Func func,
                        const float* in, const float* reference, const size_t n,
                        const size_t c, const float atol = 1e-6) {
  const size_t kElemNums = n * c;
  auto res = std::make_unique<float[]>(kElemNums);
  func(in, res.get(), n, c);
  bool success = VerifyResults(res.get(), reference, kElemNums);
  if (success) {
    std::cout << func_name << " run correctly." << std::endl;
  } else {
    std::cerr << func_name << " failed!" << std::endl;
  }
}
