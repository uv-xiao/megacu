#include <megacu/runtime.h>

namespace megacu::runtime::target {

status run_phase(std::span<linked_task const> tasks,
                 std::span<linked_event_tensor const> events,
                 progress_model progress) {
  dispatcher::dispatch_state dispatch;
  auto dispatch_status =
      dispatcher::map_explicit_attrs(dispatch, tasks, events);
  if (dispatch_status.code != status_code::ok) {
    return dispatch_status;
  }
  return scheduler::run_explicit_asap(tasks, dispatch, progress);
}

status validate_phase(std::span<linked_task const> tasks,
                      std::span<linked_event_tensor const> events,
                      progress_model progress) {
  dispatcher::dispatch_state dispatch;
  auto dispatch_status =
      dispatcher::map_explicit_attrs(dispatch, tasks, events);
  if (dispatch_status.code != status_code::ok) {
    return dispatch_status;
  }
  return scheduler::validate_explicit_asap(tasks, dispatch, progress);
}

} // namespace megacu::runtime::target

extern "C" int megacu_target_runtime_component() { return 1; }
