#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

int main() {
  const int kN = 1000;
  const size_t kSize = kN * sizeof(int);

  int* h_a = reinterpret_cast<int*>(malloc(kSize));
  int* h_b = reinterpret_cast<int*>(malloc(kSize));

  for (int i = 0; i < kN; ++i) {
    h_b[i] = i;
  }

  int* d_a;
  cudaMalloc(&d_a, kSize);

  cudaMemcpy(d_a, h_b, kSize, cudaMemcpyHostToDevice);
  cudaMemcpy(h_a, d_a, kSize, cudaMemcpyDeviceToHost);

  if (cudaGetLastError() == cudaSuccess) {
    bool success = true;
    for (int i = 0; i < kN; ++i) {
      if (h_a[i] != h_b[i]) {
        printf("wrong: wanted %d but got %d.\n", h_b[i], h_a[i]);
        success = false;
        break;
      }
    }
    if (success) {
      printf("done.\n");
    }
  }

  free(h_a);
  free(h_b);
  cudaFree(d_a);

  return 0;
}