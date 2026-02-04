#pragma once

#include <cstddef>
#include <memory>
#include <string>

std::unique_ptr<float[]> OpenTestFile(const std::string& in, const size_t size);
bool VerifyResults(const float* output, const float* reference,
                   const size_t size, const float atol = 1e-6);