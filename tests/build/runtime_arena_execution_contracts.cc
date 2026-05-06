#include <cassert>
#include <cstdint>

#include <megacu/runtime/execution/host_orch.h>

#include "runtime_recipe_contracts.h"

namespace {

bool has_shape(megacu::runtime::attr_set const &attrs, std::int64_t first,
               std::int64_t second) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::event_tensor_shape &&
        attr.first == first && attr.second == second) {
      return true;
    }
  }
  return false;
}

bool has_wait_count(megacu::runtime::attr_set const &attrs,
                    std::int64_t count) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::event_tensor_wait_count &&
        attr.first == count) {
      return true;
    }
  }
  return false;
}

bool has_tile_grid(megacu::runtime::attr_set const &attrs, std::int64_t first,
                   std::int64_t second) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::dispatch_tile_grid &&
        attr.first == first && attr.second == second) {
      return true;
    }
  }
  return false;
}

bool has_event_ref(megacu::runtime::attr_set const &attrs,
                   megacu::runtime::attr_kind kind,
                   megacu::runtime::event_tensor_ref event) {
  for (auto attr : attrs.entries()) {
    if (attr.kind == kind && attr.event == event) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() {
  megacu::runtime::execution::host_orch::frame<4, 2, 4> frame;
  auto refs = megacu::runtime::contracts::submit_three_task_event_recipe(frame);

  assert(refs.ready == megacu::runtime::event_tensor_ref{0});
  assert(refs.producer == megacu::runtime::task_ref{0});
  assert(refs.wait_ready == megacu::runtime::task_ref{1});
  assert(refs.consumer == megacu::runtime::task_ref{2});
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

  assert(events[0].ref == refs.ready);
  assert(has_shape(events[0].attributes, 1, 1));
  assert(has_wait_count(events[0].attributes, 1));

  assert(tasks[0].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[0].op_slot == megacu::runtime::operator_slot{3});
  assert(tasks[0].dep_count == 0);
  assert(has_tile_grid(tasks[0].attributes, 1, 1));
  assert(has_event_ref(tasks[0].attributes,
                       megacu::runtime::attr_kind::event_notify, refs.ready));

  assert(tasks[1].kind == megacu::runtime::task_kind::sync_only);
  assert(tasks[1].op_slot == megacu::runtime::invalid_operator_slot());
  assert(tasks[1].dep_count == 1);
  assert(deps[tasks[1].first_dep] == refs.producer);
  assert(has_event_ref(tasks[1].attributes,
                       megacu::runtime::attr_kind::event_wait, refs.ready));

  assert(tasks[2].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[2].op_slot == megacu::runtime::operator_slot{4});
  assert(tasks[2].dep_count == 1);
  assert(deps[tasks[2].first_dep] == refs.wait_ready);
  assert(has_tile_grid(tasks[2].attributes, 1, 1));

  assert(regions[0].epoch == 0);
  assert(regions[0].first_task == 0);
  assert(regions[0].task_count == 3);
  assert(regions[0].first_event == 0);
  assert(regions[0].event_count == 1);
  assert(regions[0].sealed == 1);
  return 0;
}
