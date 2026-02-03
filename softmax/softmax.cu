#include <cuda_runtime.h>

#include <chrono>
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
    max_val = fmaxf(max_val, row_in[j]);
  }

  float sum = 0.0f;
  for (size_t k = 0; k < c; ++k) {
    row_out[k] = expf(row_in[k] - max_val);
    sum += row_out[k];
  }
  sum = 1.0f / sum;

  // out
  for (size_t t = 0; t < c; ++t) {
    row_out[t] *= sum;
  }
}

// n * c, 假设每行初始化 32 个线程, 每行数据量是32的倍数
__global__ void SoftmaxV2(const float* in, float* out, const size_t n,
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

  // cpu compute part
  auto start = std::chrono::high_resolution_clock::now();
  SoftmaxCpu(data.get(), res.get(), kBlockNums, kBlockSize);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cost = end - start;
  std::cout << "Softmax Cpu costs " << cost.count() << "ms" << std::endl;

  auto data_check = OpenTestFile("expected.bin", kElemNums);
  if (data_check == nullptr) {
    std::cerr << "Read referenced datas failed." << std::endl;
    return -1;
  }
  bool cpu_checked = VerifyResults(res.get(), data_check.get(), kElemNums);
  std::cout << "cpu version check: " << (cpu_checked ? "pass!" : "fail!")
            << std::endl;
  // output reset all to 0
  for (size_t i = 0; i < kElemNums; ++i) {
    *(res.get() + i) = 0.0f;
  }

  // cuda softmax v1
  float *d_in, *d_out;
  size_t byte_size = kElemNums * sizeof(float);
  cudaMalloc(&d_in, byte_size);
  cudaMalloc(&d_out, byte_size);
  cudaMemcpy(d_in, data.get(), byte_size, cudaMemcpyHostToDevice);

  cudaEvent_t gpu_start, gpu_end;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_end);
  cudaEventRecord(gpu_start);
  SoftmaxV1<<<kBlockNums, 1>>>(d_in, d_out, kBlockNums, kBlockSize);
  cudaEventRecord(gpu_end);
  cudaEventSynchronize(gpu_end);
  float gpu_cost = 0.0f;
  cudaEventElapsedTime(&gpu_cost, gpu_start, gpu_end);
  printf("SoftmaxV1 GPU 执行时间: %f ms\n", gpu_cost);

  cudaMemcpy(res.get(), d_out, byte_size, cudaMemcpyDeviceToHost);

  bool gpu_checked1 = VerifyResults(res.get(), data_check.get(), kElemNums);
  std::cout << "gpu version1 check: " << (gpu_checked1 ? "pass!" : "fail!")
            << std::endl;

  // cuda softmax v2
  cudaMemset(d_out, 0, byte_size);

  cudaEventRecord(gpu_start);
  SoftmaxV2<<<kBlockNums, 32, 32 * sizeof(float)>>>(d_in, d_out, kBlockNums,
                                                    kBlockSize);
  cudaEventRecord(gpu_end);
  cudaEventSynchronize(gpu_end);
  gpu_cost = 0.0f;
  cudaEventElapsedTime(&gpu_cost, gpu_start, gpu_end);
  printf("SoftmaxV2 GPU 执行时间: %f ms\n", gpu_cost);

  cudaMemcpy(res.get(), d_out, byte_size, cudaMemcpyDeviceToHost);

  bool gpu_checked2 = VerifyResults(res.get(), data_check.get(), kElemNums);
  std::cout << "gpu version2 check: " << (gpu_checked2 ? "pass!" : "fail!")
            << std::endl;

  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(gpu_start);
  cudaEventDestroy(gpu_end);
  return 0;
}
