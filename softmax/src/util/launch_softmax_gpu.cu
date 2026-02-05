#include "launch_softmax_gpu_executor.cuh"
#include "scoped_timer_gpu.cuh"
#include "softmax_gpu.cuh"

void LaunchSoftmaxGpuV1(const float* in, float* out, const size_t n,
                        const size_t c) {
  LaunchSoftmaxGpuExecutor softmax_gpu_v1(SoftmaxGpuV1, "SoftmaxGpuV1", in, out,
                                          n, c);
  softmax_gpu_v1.run({n, 1}, true, 50);
}

void LaunchSoftmaxGpuV2(const float* in, float* out, const size_t n,
                        const size_t c) {
  LaunchSoftmaxGpuExecutor softmax_gpu_v2(SoftmaxGpuV2, "SoftmaxGpuV2", in, out,
                                          n, c);
  softmax_gpu_v2.run({n, 64, 64 * sizeof(float)}, true, 50);
}

void LaunchSoftmaxGpuV3(const float* in, float* out, const size_t n,
                        const size_t c) {
  LaunchSoftmaxGpuExecutor softmax_gpu_v3(SoftmaxGpuV3, "SoftmaxGpuV3", in, out,
                                          n, c);
  softmax_gpu_v3.run({n, 32}, true, 50);
}

void LaunchSoftmaxGpuV4(const float* in, float* out, const size_t n,
                        const size_t c) {
  LaunchSoftmaxGpuExecutor softmax_gpu_v4(SoftmaxGpuV4, "SoftmaxGpuV4", in, out,
                                          n, c);
  softmax_gpu_v4.run({n, 128, 128 / 32 * 2 * sizeof(float)}, true, 50);
}