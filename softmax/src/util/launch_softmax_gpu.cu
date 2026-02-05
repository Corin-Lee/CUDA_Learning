#include "launch_softmax_gpu_executor.cuh"
#include "scoped_timer_gpu.cuh"
#include "softmax_gpu.cuh"

void LaunchSoftmaxGpuV1(const float* in, float* out, const size_t n,
                        const size_t c) {
  LaunchSoftmaxGpuExecutor softmax_gpu_v1(SoftmaxGpuV1, "SoftmaxGpuV1", in, out,
                                          n, c);
  softmax_gpu_v1.run({n, 1});
}

void LaunchSoftmaxGpuV2(const float* in, float* out, const size_t n,
                        const size_t c) {
  LaunchSoftmaxGpuExecutor softmax_gpu_v2(SoftmaxGpuV2, "SoftmaxGpuV2", in, out,
                                          n, c);
  softmax_gpu_v2.run({n, 32, 32 * sizeof(float)});
}