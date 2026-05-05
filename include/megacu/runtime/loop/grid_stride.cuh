#pragma once

namespace megacu::runtime::loop {

struct grid_stride {
  template <class Context, class Arena, class Scheduler, class Dispatcher,
            class EventTensor, class Operators>
  __device__ void run(Context ctx, Arena, Scheduler scheduler,
                      Dispatcher dispatcher, EventTensor event_tensor,
                      Operators operators) const {
    for (auto work = dispatcher.first(ctx); work.active;
         work = dispatcher.next(ctx, work)) {
      if constexpr (requires { dispatcher.mark_grid_stride_work(ctx, work); }) {
        dispatcher.mark_grid_stride_work(ctx, work);
      }

      for (auto task = scheduler.first(ctx, work); task.valid();
           task = scheduler.next(ctx, work, task)) {
        event_tensor.before(ctx, task, work);
        ctx.sync_block();

        operators.invoke(ctx, task, work);
        ctx.sync_block();

        event_tensor.after(ctx, task, work);
        ctx.sync_block();

        scheduler.complete(ctx, task, work);
      }
    }
  }
};

} // namespace megacu::runtime::loop
