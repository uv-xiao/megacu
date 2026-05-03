#pragma once

namespace megacu::runtime::device {

template <class Scheduler, class Dispatcher, class Backend, class Operators>
struct entry {
  Scheduler scheduler;
  Dispatcher dispatcher;
  Backend backend;
  Operators operators;

  template <class Context> __device__ void operator()(Context ctx) const {
    for (auto work = dispatcher.first(ctx); work.active;
         work = dispatcher.next(ctx, work)) {
      for (auto task = scheduler.first(ctx, work); task.valid();
           task = scheduler.next(ctx, work, task)) {
        backend.before(ctx, task, work);
        ctx.sync_block();

        operators.invoke(ctx, task, work);
        ctx.sync_block();

        backend.after(ctx, task, work);
        ctx.sync_block();

        scheduler.complete(ctx, task, work);
      }
    }
  }
};

} // namespace megacu::runtime::device
