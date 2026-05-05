#include <cassert>
#include <cstdint>

#include <megacu/runtime/execution/host_orch.h>

namespace {

bool has_event_wait(megacu::runtime::device_task_record const &task) {
  for (auto attr : task.attributes.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::event_wait) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() {
  using megacu::runtime::operator_slot;
  namespace host_orch = megacu::runtime::execution::host_orch;

  host_orch::frame<4, 2, 4> frame;

  int event_storage[1]{};
  auto ready = frame.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(2, 3),
      megacu::runtime::event_tensor::wait_count(1),
      megacu::cuda_nvshmem::event_tensor::symmetric_storage(event_storage),
      megacu::cuda_nvshmem::event_tensor::scope::team{}));

  auto producer = frame.submit(
      operator_slot{7},
      megacu::runtime::attrs(megacu::runtime::dispatcher::tile_grid(2, 3),
                             megacu::runtime::event_tensor::notify(ready)));
  auto sync = frame.sync(megacu::runtime::attrs(
      megacu::runtime::scheduler::depends_on(producer),
      megacu::runtime::event_tensor::wait(ready)));
  frame.submit(operator_slot{11},
               megacu::runtime::attrs(
                   megacu::runtime::scheduler::depends_on(sync)));

  assert(frame.seal().code == megacu::status_code::ok);
  assert(frame.sealed());

  auto tasks = frame.tasks();
  assert(tasks.size() == 3);
  assert(tasks[0].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[0].op_slot == operator_slot{7});
  assert(tasks[0].dep_count == 0);
  assert(tasks[1].kind == megacu::runtime::task_kind::sync_only);
  assert(tasks[1].op_slot == megacu::runtime::invalid_operator_slot());
  assert(tasks[1].dep_count == 1);
  assert(has_event_wait(tasks[1]));
  assert(tasks[2].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[2].op_slot == operator_slot{11});
  assert(tasks[2].dep_count == 1);
  assert(!has_event_wait(tasks[2]));

  auto deps = frame.deps();
  assert(deps.size() == 2);
  assert(deps[tasks[1].first_dep] == producer);
  assert(deps[tasks[2].first_dep] == sync);

  auto events = frame.event_tensors();
  assert(events.size() == 1);
  assert(events[0].ref == ready);

  auto regions = frame.regions();
  assert(regions.size() == 1);
  assert(regions[0].epoch == 0);
  assert(regions[0].first_task == 0);
  assert(regions[0].task_count == 3);
  assert(regions[0].first_event == 0);
  assert(regions[0].event_count == 1);
  assert(regions[0].sealed == 1);

  auto arena = frame.arena();
  assert(arena.task_count == 3);
  assert(arena.event_count == 1);
  assert(arena.dep_count == 2);
  assert(arena.region_count == 1);
  assert(arena.task_capacity == 4);
  assert(arena.event_capacity == 2);
  assert(arena.dep_capacity == 4);
  assert(arena.region_capacity == 1);

  auto late = frame.submit(operator_slot{13});
  assert(late == megacu::runtime::invalid_task_ref());
  assert(frame.current_status().code == megacu::status_code::invalid_argument);

  host_orch::frame<2, 1, 1> dep_overflow;
  auto first = dep_overflow.submit(operator_slot{1});
  auto second = dep_overflow.submit(
      operator_slot{2},
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(first),
                             megacu::runtime::scheduler::depends_on(first)));
  assert(second == megacu::runtime::invalid_task_ref());
  assert(dep_overflow.current_status().code ==
         megacu::status_code::invalid_argument);

  return 0;
}
