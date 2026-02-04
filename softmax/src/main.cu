#include <iostream>

#include "execute_softmax_test.hpp"
#include "softmax_cpu.hpp"
#include "softmax_gpu.cuh"
#include "test.hpp"

const size_t kBlockNums = 10;
const size_t kBlockSize = 512;
const size_t kElemNums = kBlockNums * kBlockSize;

int main() {
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
  // SoftmaxCpuV1
  ExecuteSoftmaxTest("SoftmaxCpuV1", LaunchSoftmaxCpuV1, data.get(),
                     data_check.get(), kBlockNums, kBlockSize);

  ExecuteSoftmaxTest("SoftmaxGpuV1", LaunchSoftmaxGpuV1, data.get(),
                     data_check.get(), kBlockNums, kBlockSize);

  ExecuteSoftmaxTest("SoftmaxGpuV2", LaunchSoftmaxGpuV2, data.get(),
                     data_check.get(), kBlockNums, kBlockSize);
  return 0;
}
