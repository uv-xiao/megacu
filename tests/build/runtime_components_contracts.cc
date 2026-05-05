#include <cassert>
#include <cstddef>
#include <cstdint>

#include <megacu/runtime.h>

extern "C" int megacu_dispatcher_explicit_attrs_component();
extern "C" int megacu_scheduler_explicit_asap_component();
extern "C" int megacu_platform_cuda_component();
extern "C" int megacu_backend_nvshmem_component();
extern "C" int megacu_target_runtime_component();

namespace {

struct call_log {
  int first = 0;
  int second = 0;
  int observed_first_before_second = 0;
};

megacu::status first(call_log *log) {
  log->first += 1;
  return {};
}

megacu::status second(call_log *log) {
  log->second += 1;
  if (log->first == 1) {
    log->observed_first_before_second += 1;
  }
  return {};
}

struct driver {
  megacu::cuda::launch_view launch;
  megacu::nvshmem::team_view team;
};

bool has_dependency(megacu::runtime::linked_task const &task) {
  for (auto attr : task.attributes.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::dependency) {
      return true;
    }
  }
  return false;
}

bool has_event_wait(megacu::runtime::linked_task const &task) {
  for (auto attr : task.attributes.entries()) {
    if (attr.kind == megacu::runtime::attr_kind::event_wait) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() {
  assert(megacu_dispatcher_explicit_attrs_component() == 1);
  assert(megacu_scheduler_explicit_asap_component() == 1);
  assert(megacu_platform_cuda_component() == 1);
  assert(megacu_backend_nvshmem_component() == 1);
  assert(megacu_target_runtime_component() == 1);

  driver runtime_driver{};
  assert(!megacu::cuda::has_stream(runtime_driver.launch));
  runtime_driver.launch.stream = &runtime_driver;
  assert(megacu::cuda::has_stream(runtime_driver.launch));
  assert(megacu::nvshmem::team_size(runtime_driver.team) == 1);
  assert(!megacu::nvshmem::has_remote_pes(runtime_driver.team));
  runtime_driver.team.team_n_pes = 2;
  assert(megacu::nvshmem::team_size(runtime_driver.team) == 2);
  assert(megacu::nvshmem::has_remote_pes(runtime_driver.team));

  int ready[1]{};
  megacu::cuda::device::signal_ready(ready, 0, 7);
  assert(megacu::cuda::device::load_ready(ready, 0) == 7);
  assert(megacu::nvshmem::device::remote_int(ready, 0, 0) == 7);

  auto phase = megacu::runtime::make_phase(
      runtime_driver, megacu::runtime::progress_model::asap);

  call_log log;
  auto produced = phase.submit(megacu::runtime::op("first", &first), &log);
  phase.submit(megacu::runtime::op("second", &second), &log);
  phase.submit(
      megacu::runtime::op("second", &second), &log,
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(produced)));

  auto tasks = phase.tasks();
  assert(tasks.size() == 3);
  assert(!has_dependency(tasks[1]));
  assert(has_dependency(tasks[2]));

  megacu::runtime::dispatcher::dispatch_state dispatch;
  assert(
      megacu::runtime::dispatcher::map_explicit_attrs(dispatch, tasks).code ==
      megacu::status_code::ok);
  assert(dispatch.task_count == 3);
  assert(dispatch.tile_grid_attrs == 0);

  assert(phase.run().code == megacu::status_code::ok);
  assert(log.first == 1);
  assert(log.second == 2);
  assert(log.observed_first_before_second == 2);

  auto mapped_phase = megacu::runtime::make_phase(
      runtime_driver, megacu::runtime::progress_model::asap);
  mapped_phase.submit(
      megacu::runtime::op("first", &first), &log,
      megacu::runtime::attrs(megacu::runtime::dispatcher::tile_grid(2, 3)));
  megacu::runtime::dispatcher::dispatch_state mapped_dispatch;
  assert(megacu::runtime::dispatcher::map_explicit_attrs(mapped_dispatch,
                                                         mapped_phase.tasks())
             .code == megacu::status_code::ok);
  assert(mapped_dispatch.task_count == 1);
  assert(mapped_dispatch.tile_grid_attrs == 1);
  assert(mapped_dispatch.total_tile_count == 6);

  megacu::runtime::attr_set manual_attrs;
  for (std::size_t index = 0; index < megacu::runtime::attr_set::max_attrs;
       ++index) {
    assert(manual_attrs.push(megacu::runtime::scheduler::depends_on(
        megacu::runtime::task_ref{static_cast<std::uint32_t>(index)})));
  }
  assert(!manual_attrs.push(
      megacu::runtime::scheduler::depends_on(megacu::runtime::task_ref{99})));
  assert(manual_attrs.size() == megacu::runtime::attr_set::max_attrs);

  call_log sync_log;
  auto sync_phase = megacu::runtime::make_phase(
      runtime_driver, megacu::runtime::progress_model::asap);
  auto ready_events = sync_phase.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(2, 3),
      megacu::runtime::event_tensor::wait_count(1),
      megacu::cuda_nvshmem::event_tensor::symmetric_storage(ready),
      megacu::cuda_nvshmem::event_tensor::scope::team{}));
  auto sync_producer = sync_phase.submit(
      megacu::runtime::op("first", &first), &sync_log,
      megacu::runtime::attrs(
          megacu::runtime::event_tensor::notify(ready_events)));
  auto sync_only = sync_phase.sync(megacu::runtime::attrs(
      megacu::runtime::scheduler::depends_on(sync_producer),
      megacu::runtime::event_tensor::wait(ready_events)));
  sync_phase.submit(
      megacu::runtime::op("second", &second), &sync_log,
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(sync_only)));

  auto sync_tasks = sync_phase.tasks();
  assert(sync_tasks.size() == 3);
  assert(sync_tasks[1].kind == megacu::runtime::task_kind::sync_only);
  assert(sync_tasks[1].raw_arg_count == 0);
  assert(has_dependency(sync_tasks[1]));
  assert(has_event_wait(sync_tasks[1]));
  assert(has_dependency(sync_tasks[2]));
  assert(!has_event_wait(sync_tasks[2]));
  megacu::runtime::dispatcher::dispatch_state sync_dispatch;
  assert(megacu::runtime::dispatcher::map_explicit_attrs(
             sync_dispatch, sync_tasks, sync_phase.event_tensors())
             .code == megacu::status_code::ok);
  assert(sync_dispatch.sync_task_count == 1);
  assert(sync_dispatch.event_tensor_count == 1);
  assert(sync_dispatch.sync_event_count == 6);
  assert(sync_dispatch.event_notify_attrs == 1);
  assert(sync_dispatch.event_wait_attrs == 1);
  assert(sync_dispatch.event_trigger_attrs == 0);
  assert(sync_phase.run().code == megacu::status_code::ok);
  assert(sync_log.first == 1);
  assert(sync_log.second == 1);
  assert(sync_log.observed_first_before_second == 1);

  auto unsupported_phase = megacu::runtime::make_phase(
      runtime_driver, megacu::runtime::progress_model::device_persistent);
  unsupported_phase.submit(megacu::runtime::op("first", &first), &log);
  assert(unsupported_phase.run().code == megacu::status_code::unsupported);

  auto overflow_phase = megacu::runtime::phase<driver, 1>(
      runtime_driver, megacu::runtime::progress_model::asap);
  overflow_phase.submit(megacu::runtime::op("first", &first), &log);
  auto overflow_ref =
      overflow_phase.submit(megacu::runtime::op("second", &second), &log);
  assert(overflow_ref == megacu::runtime::invalid_task_ref());
  assert(overflow_phase.run().code == megacu::status_code::invalid_argument);

  return 0;
}
