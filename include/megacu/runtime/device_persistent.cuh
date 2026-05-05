#pragma once

#include <type_traits>

namespace megacu::runtime {

namespace detail {

template <class ExecutionModel, class Context, class Arena>
__device__ bool construct_should_run(ExecutionModel const &execution,
                                     Context ctx, Arena &arena) {
  if constexpr (std::is_void_v<decltype(execution.construct(ctx, arena))>) {
    execution.construct(ctx, arena);
    return true;
  } else {
    return static_cast<bool>(execution.construct(ctx, arena));
  }
}

} // namespace detail

template <class ExecutionModel, class Loop, class Scheduler, class Dispatcher,
          class EventTensor, class Operators>
struct device_persistent {
  ExecutionModel execution;
  Loop loop;
  Scheduler scheduler;
  Dispatcher dispatcher;
  EventTensor event_tensor;
  Operators operators;

  template <class Context> __device__ void operator()(Context ctx) const {
    auto arena = execution.bind(ctx);
    if (!detail::construct_should_run(execution, ctx, arena)) {
      return;
    }
    loop.run(ctx, arena, scheduler, dispatcher, event_tensor, operators);
  }
};

} // namespace megacu::runtime
