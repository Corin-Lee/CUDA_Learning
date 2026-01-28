#include <cuda_runtime.h>

#include <cstdio>

__global__ void Softmax() {}

void SoftmaxCpu(const float* in, float* out, const size_t n, const size_t c) {
  // n * c datas
  for (int i = 0; i < n; ++i) {
    const float* row_in = in + c * i;
    float* row_out = out + c * i;
    // get max value
    float max_val = *row_in;
    for (int j = 1; j < c; ++j) {
      max_val = max(max_val, row_in[j]);
    }

    float sum = 0.0f;
    for (int k = 0; k < c; ++k) {
      row_out[k] = exp(row_in[k] - max_val);
      sum += row_out[k];
    }
    sum = 1.0f / sum;

    // out
    for (int t = 0; t < c; ++t) {
      row_out[t] *= sum;
    }
  }
}

int main() { return 0; }