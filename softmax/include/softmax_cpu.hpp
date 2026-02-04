#pragma once
#include <cstddef>

void LaunchSoftmaxCpuV1(const float* in, float* out, const size_t n,
                        const size_t c);

void SoftmaxCpuV1(const float* in, float* out, const size_t n, const size_t c);
