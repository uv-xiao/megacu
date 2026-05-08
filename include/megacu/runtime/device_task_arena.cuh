#pragma once

#include <cstddef>
#include <cstdint>

#include <megacu/runtime/task_arena.h>

namespace megacu::runtime {

class device_orch {
public:
  __device__ explicit device_orch(task_arena_view &arena) : arena_(arena) {}

  __device__ event_tensor_ref event_tensor(attr_set attributes = {}) {
    if (error_.code != status_code::ok) {
      return invalid_event_ref();
    }
    if (sealed_) {
      set_error(10);
      return invalid_event_ref();
    }
    if (arena_.event_count >= arena_.event_capacity) {
      set_error(11);
      return invalid_event_ref();
    }

    auto ref = event_tensor_ref{arena_.event_count};
    arena_.events[arena_.event_count] = {.ref = ref, .attributes = attributes};
    ++arena_.event_count;
    return ref;
  }

  __device__ task_ref submit(operator_slot slot, attr_set attributes = {}) {
    return publish_task(task_kind::operator_body, slot, attributes);
  }

  __device__ task_ref sync(attr_set attributes = {}) {
    return publish_task(task_kind::sync_only, invalid_op_slot(), attributes);
  }

  __device__ status seal(std::uint32_t epoch = 0) {
    if (error_.code != status_code::ok) {
      return error_;
    }
    if (sealed_) {
      return {};
    }
    if (arena_.region_capacity == 0) {
      set_error(12);
      return error_;
    }

    arena_.regions[0] = {.epoch = epoch,
                         .first_task = 0,
                         .task_count = arena_.task_count,
                         .first_event = 0,
                         .event_count = arena_.event_count,
                         .sealed = 0};
    arena_.region_count = 1;
    __threadfence();
    static_cast<volatile arena_region *>(arena_.regions)[0].sealed = 1;
    sealed_ = true;
    return {};
  }

  __device__ status current_status() const { return error_; }

private:
  __device__ task_ref publish_task(task_kind kind, operator_slot slot,
                                   attr_set attributes) {
    if (error_.code != status_code::ok) {
      return invalid_task();
    }
    if (sealed_) {
      set_error(10);
      return invalid_task();
    }
    if (arena_.task_count >= arena_.task_capacity) {
      set_error(13);
      return invalid_task();
    }

    std::uint16_t task_dep_count = 0;
    auto entries = attributes.entries();
    for (std::size_t index = 0; index < entries.size(); ++index) {
      auto item = entries[index];
      if (item.kind != attr_kind::dependency) {
        continue;
      }
      ++task_dep_count;
    }

    auto first_dep = arena_.dep_count;
    if (arena_.dep_count > arena_.dep_capacity) {
      set_error(14);
      return invalid_task();
    }
    auto remaining_dep_capacity = arena_.dep_capacity - arena_.dep_count;
    if (task_dep_count > remaining_dep_capacity) {
      set_error(14);
      return invalid_task();
    }

    auto ref = task_ref{arena_.task_count};
    auto next_dep = first_dep;
    for (std::size_t index = 0; index < entries.size(); ++index) {
      auto item = entries[index];
      if (item.kind == attr_kind::dependency) {
        arena_.deps[next_dep++] = item.task;
      }
    }

    arena_.tasks[arena_.task_count] = {
        .kind = kind,
        .op_slot = slot,
        .dep_count = task_dep_count,
        .first_dep = static_cast<std::uint16_t>(first_dep),
        .attributes = attributes};
    initialize_task_state(ref, kind, attributes);
    arena_.dep_count = next_dep;
    ++arena_.task_count;
    return ref;
  }

  __device__ void initialize_task_state(task_ref task, task_kind kind,
                                        attr_set attributes) {
    if (arena_.task_completed != nullptr) {
      arena_.task_completed[task.value] = 0;
    }
    if (arena_.task_remaining_work != nullptr) {
      arena_.task_remaining_work[task.value] =
          initial_remaining_work(kind, attributes);
    }
  }

  __device__ std::uint32_t initial_remaining_work(task_kind kind,
                                                  attr_set attributes) const {
    if (kind == task_kind::sync_only) {
      return 1;
    }
    for (auto item : attributes.entries()) {
      if (item.kind == attr_kind::dispatch_single_tile) {
        return 1;
      }
      if (item.kind == attr_kind::dispatch_tile_grid) {
        return static_cast<std::uint32_t>(item.first * item.second);
      }
    }
    return 1;
  }

  __device__ void set_error(std::uint16_t detail) {
    error_ = {status_code::invalid_argument, detail,
              "megacu device orch arena capacity exceeded"};
  }

  __device__ static event_tensor_ref invalid_event_ref() {
    return {static_cast<std::uint32_t>(~std::uint32_t{0})};
  }

  __device__ static operator_slot invalid_op_slot() {
    return {static_cast<std::uint16_t>(~std::uint16_t{0})};
  }

  __device__ static task_ref invalid_task() {
    return {static_cast<std::uint32_t>(~std::uint32_t{0})};
  }

  task_arena_view &arena_;
  bool sealed_ = false;
  status error_;
};

} // namespace megacu::runtime
