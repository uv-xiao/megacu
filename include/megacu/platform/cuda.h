#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define MEGACU_CUDA_HOST_DEVICE __host__ __device__
#else
#define MEGACU_CUDA_HOST_DEVICE
#endif

namespace megacu::cuda {

struct launch_view {
  void *stream = nullptr;
  std::int32_t device_ordinal = 0;
};

bool has_stream(launch_view launch) noexcept;

namespace device {

MEGACU_CUDA_HOST_DEVICE inline void fence_system() {
#if defined(__CUDA_ARCH__)
  __threadfence_system();
#endif
}

MEGACU_CUDA_HOST_DEVICE inline int load_ready(
    int const *events,
    std::int64_t index) {
#if defined(__CUDA_ARCH__)
  return atomicAdd(const_cast<int *>(events + index), 0);
#else
  return events[index];
#endif
}

MEGACU_CUDA_HOST_DEVICE inline void signal_ready(
    int *events,
    std::int64_t index,
    int value) {
#if defined(__CUDA_ARCH__)
  atomicExch(events + index, value);
#else
  events[index] = value;
#endif
}

MEGACU_CUDA_HOST_DEVICE inline void wait_ready(
    int const *events,
    std::int64_t index,
    int value) {
  while (load_ready(events, index) != value) {
  }
}

}  // namespace device

}  // namespace megacu::cuda

#undef MEGACU_CUDA_HOST_DEVICE
