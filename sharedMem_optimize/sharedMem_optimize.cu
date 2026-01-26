#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <memory>

const int n = 102400;
const int blockSize = 256;

__global__ void reduceGmem(int* g_idata, int* g_odata, unsigned int n) {
  const unsigned int tid = threadIdx.x;
  int* idata = g_idata + blockIdx.x * blockDim.x;

  if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
  __syncthreads();
  if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
  __syncthreads();

  if (tid < 32) {
    volatile int* vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmem(int* s_idata, int* s_odata, unsigned int n) {
  __shared__ int smem[blockSize];
  const unsigned int tid = threadIdx.x;
  int* idata = s_idata + blockIdx.x * blockDim.x;

  smem[tid] = idata[tid];
  __syncthreads();

  if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
  __syncthreads();
  if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
  __syncthreads();

  if (tid < 32) {
    volatile int* vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  if (tid == 0) s_odata[blockIdx.x] = smem[0];
}

int main() {
  const int numBlocks = (n + blockSize - 1) / blockSize;

  std::array<int, n> h_idata;
  for (int i = 0; i < n; i++) h_idata[i] = 1;

  int *d_odata1, *d_odata2, *d_idata1, *d_idata2;
  cudaMalloc((void**)&d_idata1, n * sizeof(int));
  cudaMalloc((void**)&d_idata2, n * sizeof(int));
  cudaMalloc((void**)&d_odata1, n * sizeof(int));
  cudaMalloc((void**)&d_odata2, n * sizeof(int));

  cudaMemcpy(d_idata1, h_idata.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_idata2, h_idata.data(), n * sizeof(int), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int sum;
  cudaEventRecord(start);
  reduceGmem<<<numBlocks, blockSize>>>(d_idata1, d_odata1, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms1 = 0;
  cudaEventElapsedTime(&ms1, start, stop);
  std::cout << "Time for reduceSmem: " << ms1 << " ms" << std::endl;
  cudaMemcpy(&sum, d_odata1, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "sum1: " << sum << std::endl;

  // Measure
  cudaEventRecord(start);
  reduceSmem<<<numBlocks, blockSize>>>(d_idata2, d_odata2, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float ms2 = 0;
  cudaEventElapsedTime(&ms2, start, stop);
  std::cout << "Time for reduceSmem: " << ms2 << " ms" << std::endl;
  cudaMemcpy(&sum, d_odata2, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "sum2: " << sum << std::endl;

  cudaFree(d_idata1);
  cudaFree(d_idata2);
  cudaFree(d_odata1);
  cudaFree(d_odata2);

  return 0;
}