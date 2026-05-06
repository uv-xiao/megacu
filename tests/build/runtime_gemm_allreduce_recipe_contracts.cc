#include <cassert>

#include <megacu/runtime/execution/host_orch.h>

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh"

int main() {
  megacu::runtime::execution::host_orch::frame<4, 2, 4> frame;
  int event_storage = 0;
  auto args = gemm_ar::runtime_args{
      .driver = {},
      .a = nullptr,
      .b = nullptr,
      .partial = nullptr,
      .out = nullptr,
      .events = &event_storage,
      .problem = {.m = 2, .n = 3, .k = 4, .tile_m = 1, .tile_n = 2}};
  args.driver.team.team_n_pes = 2;

  auto result = gemm_ar::build_runtime_recipe(frame, args);
  assert(result.code == megacu::status_code::ok);
  assert(frame.seal().code == megacu::status_code::ok);

  auto tasks = frame.tasks();
  auto events = frame.event_tensors();
  auto deps = frame.deps();
  auto regions = frame.regions();

  assert(events.size() == 1);
  assert(tasks.size() == 3);
  assert(deps.size() == 2);
  assert(regions.size() == 1);
  assert(regions[0].sealed == 1);

  assert(tasks[0].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[0].op_slot == gemm_ar::runtime_slots::gemm_tile_produce);
  assert(tasks[0].dep_count == 0);

  assert(tasks[1].kind == megacu::runtime::task_kind::sync_only);
  assert(tasks[1].op_slot == megacu::runtime::invalid_operator_slot());
  assert(tasks[1].dep_count == 1);
  assert(deps[tasks[1].first_dep].value == 0);

  assert(tasks[2].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[2].op_slot == gemm_ar::runtime_slots::allreduce_tile_consume);
  assert(tasks[2].dep_count == 1);
  assert(deps[tasks[2].first_dep].value == 1);

  bool event_has_shape = false;
  bool event_has_wait_count = false;
  bool event_has_storage = false;
  bool event_has_scope = false;
  for (auto attr : events[0].attributes.entries()) {
    event_has_shape |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_shape &&
        attr.first == 2 && attr.second == 2;
    event_has_wait_count |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_wait_count &&
        attr.first == 2;
    event_has_storage |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_storage &&
        attr.pointer == &event_storage;
    event_has_scope |=
        attr.kind == megacu::runtime::attr_kind::event_tensor_scope;
  }

  bool task0_has_dispatch = false;
  bool task0_has_notify = false;
  bool task1_has_wait = false;
  bool task2_has_dispatch = false;
  for (auto attr : tasks[0].attributes.entries()) {
    task0_has_dispatch |=
        attr.kind == megacu::runtime::attr_kind::dispatch_tile_grid &&
        attr.first == 2 && attr.second == 2;
    task0_has_notify |=
        attr.kind == megacu::runtime::attr_kind::event_notify &&
        attr.event.value == 0;
  }
  for (auto attr : tasks[1].attributes.entries()) {
    task1_has_wait |= attr.kind == megacu::runtime::attr_kind::event_wait &&
                      attr.event.value == 0;
  }
  for (auto attr : tasks[2].attributes.entries()) {
    task2_has_dispatch |=
        attr.kind == megacu::runtime::attr_kind::dispatch_tile_grid &&
        attr.first == 2 && attr.second == 2;
  }
  assert(event_has_shape && event_has_wait_count && event_has_storage &&
         event_has_scope);
  assert(task0_has_dispatch && task0_has_notify);
  assert(task1_has_wait);
  assert(task2_has_dispatch);
  return 0;
}
