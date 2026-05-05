#pragma once

#include <cstdint>

#include <megacu/platform/cuda.h>

#if defined(MEGACU_HAS_DEVICE_NVSHMEM)
#include <nvshmem.h>
#endif

namespace megacu::backend::nvshmem::cuda {

struct event_tensor_i32 {
  int *events = nullptr;
  std::int64_t tiles = 0;
  int my_pe = 0;
  int n_pes = 1;

  __device__ void notify(std::int64_t tile_id, int ready_value) const {
    megacu::cuda::device::fence_system();
    auto slot = static_cast<std::int64_t>(my_pe) * tiles + tile_id;
    for (int pe = 0; pe < n_pes; ++pe) {
      if (pe == my_pe) {
        megacu::cuda::device::signal_ready(events, slot, ready_value);
      } else {
#if defined(MEGACU_HAS_DEVICE_NVSHMEM)
        nvshmem_int_p(events + slot, ready_value, pe);
#endif
      }
    }
  }

  __device__ void wait(std::int64_t tile_id, int ready_value) const {
    for (int pe = 0; pe < n_pes; ++pe) {
      auto slot = static_cast<std::int64_t>(pe) * tiles + tile_id;
      megacu::cuda::device::wait_ready(events, slot, ready_value);
    }
  }
};

struct task_event_tensor_i32 {
  event_tensor_i32 event_tensor;
  int notify_task = -1;
  int wait_task = -1;

  template <class Context, class Task, class Work>
  __device__ void before(Context, Task task, Work work) const {
    if (task.value == wait_task && threadIdx.x == 0) {
      event_tensor.wait(work.tile_id, static_cast<int>(work.tile_id + 1));
    }
  }

  template <class Context, class Task, class Work>
  __device__ void after(Context, Task task, Work work) const {
    if (task.value == notify_task && threadIdx.x == 0) {
      event_tensor.notify(work.tile_id, static_cast<int>(work.tile_id + 1));
    }
  }
};

} // namespace megacu::backend::nvshmem::cuda
