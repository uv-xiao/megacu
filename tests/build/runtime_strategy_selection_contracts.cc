#include <cstdio>
#include <type_traits>

#ifndef __CUDACC__
#define __device__
#endif

#include <megacu/runtime/loop/block_tile.cuh>
#include <megacu/runtime/loop/grid_stride.cuh>
#include <megacu/runtime/loop/single_work_item.cuh>

namespace {

struct fake_counters {
  int dispatcher_first_calls = 0;
  int dispatcher_next_calls = 0;
  int grid_stride_work_marks = 0;
  int scheduler_first_calls = 0;
  int scheduler_next_calls = 0;
  int scheduler_complete_calls = 0;
  int event_before_calls = 0;
  int event_after_calls = 0;
  int operator_invocations = 0;
  int sync_calls = 0;
};

struct fake_context {
  fake_counters *counters = nullptr;

  void sync_block() {
    ++counters->sync_calls;
  }
};

struct fake_arena {};

struct fake_work {
  int index = 0;
  bool active = false;
};

struct fake_task {
  int index = 0;
  bool active = false;

  bool valid() const {
    return active;
  }
};

struct fake_scheduler {
  fake_counters *counters = nullptr;

  fake_task first(fake_context, fake_work) const {
    ++counters->scheduler_first_calls;
    return fake_task{0, true};
  }

  fake_task next(fake_context, fake_work, fake_task task) const {
    ++counters->scheduler_next_calls;
    return fake_task{task.index + 1, false};
  }

  void complete(fake_context, fake_task, fake_work) const {
    ++counters->scheduler_complete_calls;
  }
};

struct fake_dispatcher {
  fake_counters *counters = nullptr;
  int work_count = 0;

  fake_work first(fake_context) const {
    ++counters->dispatcher_first_calls;
    return fake_work{0, work_count > 0};
  }

  fake_work next(fake_context, fake_work work) const {
    ++counters->dispatcher_next_calls;
    auto next_index = work.index + 1;
    return fake_work{next_index, next_index < work_count};
  }

  void mark_grid_stride_work(fake_context, fake_work) const {
    ++counters->grid_stride_work_marks;
  }
};

struct fake_event_tensor {
  fake_counters *counters = nullptr;

  void before(fake_context, fake_task, fake_work) const {
    ++counters->event_before_calls;
  }

  void after(fake_context, fake_task, fake_work) const {
    ++counters->event_after_calls;
  }
};

struct fake_operators {
  fake_counters *counters = nullptr;

  void invoke(fake_context, fake_task, fake_work) const {
    ++counters->operator_invocations;
  }
};

template <class Loop>
constexpr bool has_runtime_loop_run() {
  return requires(Loop loop, fake_context ctx, fake_arena arena,
                  fake_scheduler scheduler, fake_dispatcher dispatcher,
                  fake_event_tensor event_tensor, fake_operators operators) {
    loop.run(ctx, arena, scheduler, dispatcher, event_tensor, operators);
  };
}

template <class Loop>
fake_counters instantiate_loop_body(int work_count) {
  fake_counters counters;
  Loop loop;
  loop.run(fake_context{&counters}, fake_arena{},
           fake_scheduler{&counters},
           fake_dispatcher{&counters, work_count},
           fake_event_tensor{&counters}, fake_operators{&counters});
  return counters;
}

bool expect_eq(const char *name, int actual, int expected) {
  if (actual == expected) {
    return true;
  }
  std::fprintf(stderr, "%s: expected %d, got %d\n", name, expected, actual);
  return false;
}

} // namespace

int main() {
  static_assert(std::is_empty_v<megacu::runtime::loop::block_tile>);
  static_assert(std::is_empty_v<megacu::runtime::loop::grid_stride>);
  static_assert(std::is_empty_v<megacu::runtime::loop::single_work_item>);
  static_assert(has_runtime_loop_run<megacu::runtime::loop::block_tile>());
  static_assert(has_runtime_loop_run<megacu::runtime::loop::grid_stride>());
  static_assert(
      has_runtime_loop_run<megacu::runtime::loop::single_work_item>());

  auto block_tile =
      instantiate_loop_body<megacu::runtime::loop::block_tile>(3);
  auto grid_stride =
      instantiate_loop_body<megacu::runtime::loop::grid_stride>(3);
  auto single_work_item =
      instantiate_loop_body<megacu::runtime::loop::single_work_item>(3);

  bool ok = true;
  ok &= expect_eq("block_tile operator invocations",
                  block_tile.operator_invocations, 3);
  ok &= expect_eq("block_tile grid-stride work marks",
                  block_tile.grid_stride_work_marks, 0);
  ok &= expect_eq("grid_stride operator invocations",
                  grid_stride.operator_invocations, 3);
  ok &= expect_eq("grid_stride work marks",
                  grid_stride.grid_stride_work_marks, 3);
  ok &= expect_eq("single_work_item operator invocations",
                  single_work_item.operator_invocations, 1);
  ok &= expect_eq("single_work_item dispatcher next calls",
                  single_work_item.dispatcher_next_calls, 0);

  if (!ok) {
    return 1;
  }
  return 0;
}
