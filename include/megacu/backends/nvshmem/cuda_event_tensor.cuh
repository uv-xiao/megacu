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
  std::int64_t event_count = 1;
  std::int64_t tiles = 0;
  int my_pe = 0;
  int n_pes = 1;

  __device__ std::int64_t slot(megacu::runtime::event_tensor_ref event, int pe,
                               std::int64_t tile_id) const {
    return (static_cast<std::int64_t>(event.value) * n_pes + pe) * tiles +
           tile_id;
  }

  __device__ void notify(megacu::runtime::event_tensor_ref event,
                         std::int64_t tile_id, int ready_value) const {
    megacu::cuda::device::fence_system();
    auto const event_slot = slot(event, my_pe, tile_id);
    for (int pe = 0; pe < n_pes; ++pe) {
      if (pe == my_pe) {
        megacu::cuda::device::signal_ready(events, event_slot, ready_value);
      } else {
#if defined(MEGACU_HAS_DEVICE_NVSHMEM)
        nvshmem_int_p(events + event_slot, ready_value, pe);
#endif
      }
    }
  }

  __device__ void wait(megacu::runtime::event_tensor_ref event,
                       std::int64_t tile_id, int ready_value) const {
    for (int pe = 0; pe < n_pes; ++pe) {
      auto const event_slot = slot(event, pe, tile_id);
      megacu::cuda::device::wait_ready(events, event_slot, ready_value);
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
          if (attr.first >= 0) {
            wait_tiles(attr);
          } else {
            wait_event(arena, attr.event);
          }
        }
      }
    }
    return true;
  }

  template <class Context, class Arena, class Task, class Work>
  __device__ void after(Context ctx, Arena arena, Task task, Work work) const {
    if (!task.valid() || arena.tasks == nullptr ||
        task.value >= static_cast<int>(arena.task_count)) {
      return;
    }

    auto const attrs = arena.tasks[task.value].attributes.entries();
    if (is_cta_leader(ctx)) {
      for (auto const &attr : attrs) {
        if (attr.kind == megacu::runtime::attr_kind::event_notify) {
          event_tensor.notify(attr.event, work.tile_id,
                              ready_value(work.tile_id));
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

  template <class Arena>
  __device__ void wait_event(Arena arena,
                             megacu::runtime::event_tensor_ref event) const {
    auto const tiles = event_tile_count(arena, event);
    for (std::int64_t tile_id = 0; tile_id < tiles; ++tile_id) {
      event_tensor.wait(event, tile_id, ready_value(tile_id));
    }
  }

  __device__ void wait_tiles(megacu::runtime::attr attr) const {
    auto const count = attr.second <= 0 ? 1 : attr.second;
    auto const stride = attr.third == 0 ? 1 : attr.third;
    for (std::int64_t offset = 0; offset < count; ++offset) {
      auto const tile_id = attr.first + offset * stride;
      event_tensor.wait(attr.event, tile_id, ready_value(tile_id));
    }
  }

  template <class Arena>
  __device__ std::int64_t
  event_tile_count(Arena arena, megacu::runtime::event_tensor_ref event) const {
    if (arena.events == nullptr ||
        event.value >= static_cast<std::uint32_t>(arena.event_count)) {
      return 1;
    }

    auto const attrs = arena.events[event.value].attributes.entries();
    for (auto const &attr : attrs) {
      if (attr.kind == megacu::runtime::attr_kind::event_tensor_shape) {
        return attr.first * attr.second;
      }
    }
    return 1;
  }

  __device__ int ready_value(std::int64_t tile_id) const {
    return static_cast<int>(tile_id + 1);
  }
};

} // namespace megacu::backend::nvshmem::cuda
