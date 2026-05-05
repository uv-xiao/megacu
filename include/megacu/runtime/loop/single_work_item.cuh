#pragma once

namespace megacu::runtime::loop {

struct single_work_item {
  template <class Context, class Arena, class Scheduler, class Dispatcher,
            class EventTensor, class Operators>
  __device__ void run(Context ctx, Arena, Scheduler scheduler,
                      Dispatcher dispatcher, EventTensor event_tensor,
                      Operators operators) const {
    auto work = dispatcher.first(ctx);
    if (!work.active) {
      return;
    }

    auto task = scheduler.first(ctx, work);
    if (!task.valid()) {
      return;
    }

    event_tensor.before(ctx, task, work);
    ctx.sync_block();

    operators.invoke(ctx, task, work);
    ctx.sync_block();

    event_tensor.after(ctx, task, work);
    ctx.sync_block();

    scheduler.complete(ctx, task, work);
  }
};

} // namespace megacu::runtime::loop
