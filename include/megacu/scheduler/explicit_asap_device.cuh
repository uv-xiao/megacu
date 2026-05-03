#pragma once

#include <megacu/runtime/device_types.cuh>

namespace megacu::scheduler::device {

struct explicit_asap {
  int task_count = 0;

  template <class Context, class Work>
  __device__ megacu::runtime::device::task_ref first(Context, Work) const {
    if (task_count <= 0) {
      return {};
    }
    return {.value = 0};
  }

  template <class Context, class Work>
  __device__ megacu::runtime::device::task_ref
  next(Context, Work, megacu::runtime::device::task_ref current) const {
    auto next_task = current.value + 1;
    if (next_task >= task_count) {
      return {};
    }
    return {.value = next_task};
  }

  template <class Context, class Work>
  __device__ void complete(Context, megacu::runtime::device::task_ref,
                           Work) const {}
};

} // namespace megacu::scheduler::device
