#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

const int kBlockSize = 256;

__global__ void vecAdd(const int* A, const int* B, int* C, const int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  const int kN = 100000;
  const size_t kSize = kN * sizeof(int);

  int* A = (int*)malloc(kSize);
  int* B = (int*)malloc(kSize);
  int* C = (int*)malloc(kSize);

  for (int i = 0; i < kN; ++i) {
    A[i] = i;
    B[i] = 2 * i;
  }

  int *d_A, *d_B;
  int* d_C;

  cudaMalloc(reinterpret_cast<void**>(&d_A), kSize);
  cudaMalloc(reinterpret_cast<void**>(&d_B), kSize);
  cudaMalloc(reinterpret_cast<void**>(&d_C), kSize);

  cudaMemcpy(d_A, A, kSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, kSize, cudaMemcpyHostToDevice);

  const int numBlocks = (kN + kBlockSize - 1) / kBlockSize;

  vecAdd<<<numBlocks, kBlockSize>>>(d_A, d_B, d_C, kN);
  cudaDeviceSynchronize();
  cudaMemcpy(C, d_C, kSize, cudaMemcpyDeviceToHost);

  // check result
  if (cudaGetLastError() == cudaSuccess) {
    bool success = true;
    for (int i = 0; i < kN; ++i) {
      if (C[i] != A[i] + B[i]) {
        printf("Error at index %d: want %d, get %d\n", i, A[i] + B[i], C[i]);
        success = false;
        break;
      }
    }
    if (success) {
      printf("addVec runs correctly.\n");
    }
  }

  free(A);
  free(B);
  free(C);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
