#pragma once

#include <megacu/runtime/device_types.cuh>
#include <megacu/runtime/task_arena.h>

namespace megacu::runtime::loop {

struct block_tile {
  template <class Context, class Arena, class Scheduler, class Dispatcher,
            class EventTensor, class Operators>
  __device__ void run(Context ctx, Arena arena, Scheduler scheduler,
                      Dispatcher dispatcher, EventTensor event_tensor,
                      Operators operators) const {
    if constexpr (requires(Context c, Arena a, Scheduler s, Dispatcher d,
                           EventTensor e, Operators o,
                           megacu::runtime::device::task_ref task,
                           megacu::runtime::device::work_item work) {
                    s.first(c, a, work);
                    s.next(c, a, work, task);
                    s.complete(c, a, task, work);
                    d.first(c, a, task);
                    d.next(c, a, task, work);
                    e.complete(c, a, task, work);
                    e.after(c, a, task, work);
                    requires requires { o.invoke(c, a, task, work); } ||
                                 requires { o.invoke(c, task, work); };
                  }) {
      run_arena(ctx, arena, scheduler, dispatcher, event_tensor, operators);
    } else {
      run_legacy(ctx, scheduler, dispatcher, event_tensor, operators);
    }
  }

private:
  template <class Context, class Scheduler, class Dispatcher, class EventTensor,
            class Operators>
  __device__ void run_legacy(Context ctx, Scheduler scheduler,
                             Dispatcher dispatcher, EventTensor event_tensor,
                             Operators operators) const {
    for (auto work = dispatcher.first(ctx); work.active;
         work = dispatcher.next(ctx, work)) {
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

  template <class Context, class Arena, class Scheduler, class Dispatcher,
            class EventTensor, class Operators>
  __device__ void run_arena(Context ctx, Arena arena, Scheduler scheduler,
                            Dispatcher dispatcher, EventTensor event_tensor,
                            Operators operators) const {
    wait_for_region_zero(arena);

    auto probe_work = megacu::runtime::device::work_item{};
    while (!region_zero_complete(arena)) {
      bool progressed = false;

      for (auto task = scheduler.first(ctx, arena, probe_work); task.valid();
           task = scheduler.next(ctx, arena, probe_work, task)) {
        if (arena.tasks == nullptr ||
            task.value >= static_cast<int>(arena.task_count)) {
          continue;
        }

        auto const &record = arena.tasks[task.value];
        if (record.kind == megacu::runtime::task_kind::sync_only) {
          auto cursor = dispatcher.first(ctx, arena, task);
          if (cursor.active) {
            auto const complete = event_tensor.complete(ctx, arena, task, cursor);
            sync_block_if_available(ctx);
            if (complete && is_cta_leader(ctx)) {
              scheduler.complete(ctx, arena, task, cursor);
            }
            sync_block_if_available(ctx);
            progressed = complete;
          }
          continue;
        }

        auto task_had_work = false;
        for (auto cursor = dispatcher.first(ctx, arena, task); cursor.active;
             cursor = dispatcher.next(ctx, arena, task, cursor)) {
          task_had_work = true;
          invoke_operator(operators, ctx, arena, task, cursor);
          sync_block_if_available(ctx);
          event_tensor.after(ctx, arena, task, cursor);
          sync_block_if_available(ctx);
          if (is_cta_leader(ctx)) {
            scheduler.complete(ctx, arena, task, cursor);
          }
          sync_block_if_available(ctx);
          progressed = true;
        }

        if (task_had_work) {
          wait_for_task_completion(arena, task);
        }
      }

      if (!progressed) {
        publish_completion();
      }
    }
  }

  template <class Arena> __device__ void wait_for_region_zero(Arena arena) const {
    if (arena.regions == nullptr || arena.region_count == 0) {
      return;
    }
    while (load_volatile(&arena.regions[0].sealed) == 0) {
    }
  }

  template <class Arena> __device__ bool region_zero_complete(Arena arena) const {
    if (arena.regions == nullptr || arena.region_count == 0) {
      return true;
    }
    if (arena.task_completed == nullptr) {
      return true;
    }

    auto const first_task = arena.regions[0].first_task;
    auto const one_past_last = first_task + arena.regions[0].task_count;
    for (auto task = first_task; task < one_past_last && task < arena.task_count;
         ++task) {
      if (load_volatile(&arena.task_completed[task]) == 0) {
        return false;
      }
    }
    return true;
  }

  template <class Arena, class Task>
  __device__ void wait_for_task_completion(Arena arena, Task task) const {
    if (arena.task_completed == nullptr || !task.valid() ||
        task.value >= static_cast<int>(arena.task_count)) {
      return;
    }
    while (load_volatile(&arena.task_completed[task.value]) == 0) {
    }
  }

  template <class Context> __device__ bool is_cta_leader(Context ctx) const {
    if constexpr (requires { ctx.thread_id(); }) {
      return ctx.thread_id() == 0;
    } else {
      return true;
    }
  }

  template <class Context>
  __device__ void sync_block_if_available(Context ctx) const {
    if constexpr (requires { ctx.sync_block(); }) {
      ctx.sync_block();
    }
  }

  template <class T> __device__ T load_volatile(T const *ptr) const {
    return *reinterpret_cast<T const volatile *>(ptr);
  }

  template <class Operators, class Context, class Arena, class Task, class Work>
  __device__ void invoke_operator(Operators operators, Context ctx, Arena arena,
                                  Task task, Work work) const {
    if constexpr (requires { operators.invoke(ctx, arena, task, work); }) {
      operators.invoke(ctx, arena, task, work);
    } else {
      operators.invoke(ctx, task, work);
    }
  }

  __device__ void publish_completion() const {
#if defined(__CUDA_ARCH__)
    __threadfence();
#endif
  }
};

} // namespace megacu::runtime::loop
