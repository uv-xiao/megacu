#pragma once

#include <megacu/runtime/loop/block_tile.cuh>
#include <megacu/runtime/task_arena.h>

namespace megacu::runtime::device {

template <class Scheduler, class Dispatcher, class EventTensor, class Operators>
struct block_tile_runtime {
  Scheduler scheduler;
  Dispatcher dispatcher;
  EventTensor event_tensor;
  Operators operators;

  template <class Context> __device__ void operator()(Context ctx) const {
    megacu::runtime::loop::block_tile{}.run(
        ctx, megacu::runtime::task_arena_view{}, scheduler, dispatcher,
        event_tensor, operators);
  }
};

} // namespace megacu::runtime::device
