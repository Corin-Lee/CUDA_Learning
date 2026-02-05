#include <cstdlib>
#include <iostream>
#include <string>

#include "execute_softmax_test.hpp"
#include "softmax_cpu.hpp"
#include "softmax_gpu.cuh"
#include "test.hpp"

// const size_t kN = 10;
// const size_t kC = 512;
const size_t kN = 1024;
const size_t kC = 1024;
const size_t kElemNums = kN * kC;

int main() {
  // 0. Create test datas
  const std::string py_cmd = "python3 ../test_data/test_data_gen.py " +
                             std::to_string(kN) + " " + std::to_string(kC) +
                             " ../test_data/";
  if (std::system(py_cmd.data())) {
    std::cerr << "Create test datas failed!" << std::endl;
    return -1;
  }

  // 1. Open test datas
  auto data = OpenTestFile("../test_data/input.bin", kElemNums);
  if (data == nullptr) {
    std::cerr << "Read input datas failed." << std::endl;
    return -1;
  }
  auto data_check = OpenTestFile("../test_data/expected.bin", kElemNums);
  if (data_check == nullptr) {
    std::cerr << "Read referenced datas failed." << std::endl;
    return -1;
  }
  // 2. run softmax
  ExecuteSoftmaxTest("SoftmaxCpuV1", LaunchSoftmaxCpuV1, data.get(),
                     data_check.get(), kN, kC);

  ExecuteSoftmaxTest("SoftmaxGpuV1", LaunchSoftmaxGpuV1, data.get(),
                     data_check.get(), kN, kC);

  ExecuteSoftmaxTest("SoftmaxGpuV2", LaunchSoftmaxGpuV2, data.get(),
                     data_check.get(), kN, kC);

  ExecuteSoftmaxTest("SoftmaxGpuV3", LaunchSoftmaxGpuV3, data.get(),
                     data_check.get(), kN, kC);
  return 0;
}
