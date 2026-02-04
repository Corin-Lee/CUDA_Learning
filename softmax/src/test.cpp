#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

std::unique_ptr<float[]> OpenTestFile(const std::string& in,
                                      const size_t size) {
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

bool VerifyResults(const float* output, const float* reference,
                   const size_t size, const float atol = 1e-6) {
  float max_error = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    max_error = std::fmax(std::fabs(output[i] - reference[i]), max_error);
  }
  printf("Max error: %e\n", max_error);
  return max_error < atol;
}