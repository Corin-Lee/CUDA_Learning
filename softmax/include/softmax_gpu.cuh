#include <cstddef>

// n个block， 每个block 1个线程
__global__ void SoftmaxV1(const float* in, float* out, const size_t n,
                          const size_t c);

// n * c, 假设每行初始化 32 个线程, 每行数据量是32的倍数
__global__ void SoftmaxV2(const float* in, float* out, const size_t n,
                          const size_t c);