#pragma once
#include <cuda_runtime.h>

#include <cstddef>
void LaunchSoftmaxGpuV1(const float* in, float* out, const size_t n,
                        const size_t c);
void LaunchSoftmaxGpuV2(const float* in, float* out, const size_t n,
                        const size_t c);
void LaunchSoftmaxGpuV3(const float* in, float* out, const size_t n,
                        const size_t c);

__global__ void SoftmaxGpuV1(const float* in, float* out, const size_t n,
                             const size_t c);

__global__ void SoftmaxGpuV2(const float* in, float* out, const size_t n,
                             const size_t c);

__global__ void SoftmaxGpuV3(const float* in, float* out, const size_t n,
                             const size_t c);