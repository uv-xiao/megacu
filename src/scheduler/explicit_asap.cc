#include <cstddef>
#include <cstdint>
#include <vector>

#include <megacu/runtime.h>

namespace {

bool dependencies_completed(
    megacu::runtime::linked_task const &task,
    std::vector<std::uint8_t> const &completed) {
  for (auto attr : task.attributes.entries()) {
    if (attr.kind != megacu::runtime::attr_kind::dependency) {
      continue;
    }
    if (attr.task.value >= completed.size() || completed[attr.task.value] == 0) {
      return false;
    }
  }
  return true;
}

megacu::status complete_sync_task(
    megacu::runtime::dispatcher::dispatch_state const &dispatch) {
  (void)dispatch;
  return {};
}

megacu::status visit_explicit_asap(
    std::span<megacu::runtime::linked_task const> tasks,
    megacu::runtime::dispatcher::dispatch_state const &dispatch,
    megacu::runtime::progress_model progress,
    bool invoke_operator_tasks) {
  if (progress != megacu::runtime::progress_model::asap) {
    return {
        megacu::status_code::unsupported,
        1,
        "explicit ASAP scheduler only supports ASAP progress"};
  }

  std::vector<std::uint8_t> completed(tasks.size(), 0);
  std::size_t completed_count = 0;

  while (completed_count < tasks.size()) {
    std::size_t selected = tasks.size();

    for (std::size_t index = 0; index < tasks.size(); ++index) {
      if (completed[index] != 0 ||
          !dependencies_completed(tasks[index], completed)) {
        continue;
      }

      selected = index;
      break;
    }

    if (selected == tasks.size()) {
      return {
          megacu::status_code::invalid_argument,
          2,
          "task dependency cycle or invalid dependency"};
    }

    if (tasks[selected].kind == megacu::runtime::task_kind::sync_only) {
      auto sync_status = complete_sync_task(dispatch);
      if (sync_status.code != megacu::status_code::ok) {
        return sync_status;
      }
      completed[selected] = 1;
      ++completed_count;
      continue;
    }

    if (invoke_operator_tasks) {
      if (!tasks[selected].invoke) {
        return {
            megacu::status_code::invalid_argument,
            3,
            "missing task callable"};
      }

      auto task_status = tasks[selected].invoke();
      if (task_status.code != megacu::status_code::ok) {
        return task_status;
      }
    }

    completed[selected] = 1;
    ++completed_count;
  }

  return {};
}

}  // namespace

namespace megacu::runtime::scheduler {

status run_explicit_asap(
    std::span<linked_task const> tasks,
    dispatcher::dispatch_state const &dispatch,
    progress_model progress) {
  return visit_explicit_asap(tasks, dispatch, progress, true);
}

status validate_explicit_asap(
    std::span<linked_task const> tasks,
    dispatcher::dispatch_state const &dispatch,
    progress_model progress) {
  return visit_explicit_asap(tasks, dispatch, progress, false);
}

}  // namespace megacu::runtime::scheduler

extern "C" int megacu_scheduler_explicit_asap_component() {
  return 1;
}
