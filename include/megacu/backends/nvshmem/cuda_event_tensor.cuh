#pragma once

#include <cstdint>

#include <megacu/platform/cuda.h>
#include <megacu/runtime.h>

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

struct attr_event_tensor_i32 {
  event_tensor_i32 event_tensor;

  template <class Context, class Arena, class Task, class Work>
  __device__ bool complete(Context ctx, Arena arena, Task task,
                           Work) const {
    if (!task.valid() || arena.tasks == nullptr ||
        task.value >= static_cast<int>(arena.task_count)) {
      return true;
    }

    auto const attrs = arena.tasks[task.value].attributes.entries();
    if (is_cta_leader(ctx)) {
      for (auto const &attr : attrs) {
        if (attr.kind == megacu::runtime::attr_kind::event_wait) {
          event_tensor.wait(event_tile(attr.event), ready_value(attr.event));
        }
      }
    }
    return true;
  }

  template <class Context, class Arena, class Task, class Work>
  __device__ void after(Context ctx, Arena arena, Task task, Work) const {
    if (!task.valid() || arena.tasks == nullptr ||
        task.value >= static_cast<int>(arena.task_count)) {
      return;
    }

    auto const attrs = arena.tasks[task.value].attributes.entries();
    if (is_cta_leader(ctx)) {
      for (auto const &attr : attrs) {
        if (attr.kind == megacu::runtime::attr_kind::event_notify) {
          event_tensor.notify(event_tile(attr.event), ready_value(attr.event));
        }
      }
    }
  }

private:
  template <class Context> __device__ bool is_cta_leader(Context ctx) const {
    if constexpr (requires { ctx.thread_id(); }) {
      return ctx.thread_id() == 0;
    } else {
      return true;
    }
  }

  __device__ std::int64_t
  event_tile(megacu::runtime::event_tensor_ref event) const {
    return static_cast<std::int64_t>(event.value);
  }

  __device__ int ready_value(megacu::runtime::event_tensor_ref event) const {
    return static_cast<int>(event_tile(event) + 1);
  }
};

} // namespace megacu::backend::nvshmem::cuda
