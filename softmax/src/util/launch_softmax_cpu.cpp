#include "scoped_timer_cpu.hpp"
#include "softmax_cpu.hpp"

void LaunchSoftmaxCpuV1(const float* in, float* out, const size_t n,
                        const size_t c) {
  ScopedTimerCpu("SoftmaxCpuV1");
  SoftmaxCpuV1(in, out, n, c);
}