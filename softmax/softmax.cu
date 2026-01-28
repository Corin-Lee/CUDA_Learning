#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

// 最朴素版本
__global__ void SoftmaxV1(const float* in, float* out, const size_t n,
                          const size_t c) {
  // n个block， 每个block 1个线程
  const size_t i = blockIdx.x;
  const float* row_in = in + c * i;
  float* row_out = out + c * i;
  // get max value
  float max_val = row_in[0];
  for (size_t j = 1; j < c; ++j) {
    max_val = std::fmax(max_val, row_in[j]);
  }

  float sum = 0.0f;
  for (size_t k = 0; k < c; ++k) {
    row_out[k] = std::exp(row_in[k] - max_val);
    sum += row_out[k];
  }
  sum = 1.0f / sum;

  // out
  for (size_t t = 0; t < c; ++t) {
    row_out[t] *= sum;
  }
}

std::unique_ptr<float[]> OpenTestFile(const std::string& in, size_t size) {
  std::ifstream data(in, std::ios::binary);
  if (!data) {
    std::cerr << "Failed to open file: " << in << std::endl;
    return nullptr;
  }
  auto res = std::make_unique<float[]>(size);
  auto& success =
      data.read(reinterpret_cast<char*>(res.get()), size * sizeof(float));
  if (!success) {
    std::cerr << "Failed to read file: " << in << std::endl;
    return nullptr;
  }
  return res;
}

bool VerifyResults(const float* output, const float* reference, size_t size,
                   const float atol = 1e-6) {
  float max_error = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    max_error = std::fmax(std::fabs(output[i] - reference[i]), max_error);
  }
  printf("Max error: %e\n", max_error);
  return max_error < atol;
}

void SoftmaxCpu(const float* in, float* out, const size_t n, const size_t c) {
  // n * c datas
  for (size_t i = 0; i < n; ++i) {
    const float* row_in = in + c * i;
    float* row_out = out + c * i;
    // get max value
    float max_val = row_in[0];
    for (size_t j = 1; j < c; ++j) {
      max_val = std::fmax(max_val, row_in[j]);
    }

    float sum = 0.0f;
    for (size_t k = 0; k < c; ++k) {
      row_out[k] = std::exp(row_in[k] - max_val);
      sum += row_out[k];
    }
    sum = 1.0f / sum;

    // out
    for (size_t t = 0; t < c; ++t) {
      row_out[t] *= sum;
    }
  }
}

const size_t kBlockNums = 10;
const size_t kBlockSize = 512;

int main() {
  const size_t kElemNums = kBlockNums * kBlockSize;
  auto data = OpenTestFile("input.bin", kElemNums);
  if (data == nullptr) {
    std::cerr << "Read input datas failed." << std::endl;
    return -1;
  }

  auto res = std::make_unique<float[]>(kElemNums);
  SoftmaxCpu(data.get(), res.get(), kBlockNums, kBlockSize);
  auto data_check = OpenTestFile("expected.bin", kElemNums);
  if (data_check == nullptr) {
    std::cerr << "Read referenced datas failed." << std::endl;
    return -1;
  }
  bool cpu_checked = VerifyResults(res.get(), data_check.get(), kElemNums);
  std::cout << "cpu version check: " << (cpu_checked ? "pass!" : "fail!")
            << std::endl;

  SoftmaxV1<<<kElemNums, 1>>>(data.get(), res.get(), kBlockNums, kBlockSize);
  bool gpu_checked1 = VerifyResults(res.get(), data_check.get(), kElemNums);
  std::cout << "gpu version1 check: " << (gpu_checked1 ? "pass!" : "fail!")
            << std::endl;

  return 0;
}
