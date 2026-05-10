#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <megacu/runtime/task_arena.h>

#if defined(__CUDACC__)
#define MEGACU_HOST_ORCH_HOST_DEVICE __host__ __device__
#else
#define MEGACU_HOST_ORCH_HOST_DEVICE
#endif

namespace megacu::runtime::execution::host_orch {

struct model {
  task_arena_view arena;

  template <class Context>
  MEGACU_HOST_ORCH_HOST_DEVICE task_arena_view bind(Context) const {
    return arena;
  }

  template <class Context>
  MEGACU_HOST_ORCH_HOST_DEVICE bool construct(Context, task_arena_view &) const {
    return true;
  }
};

template <std::size_t MaxTasks = 64, std::size_t MaxEvents = 16,
          std::size_t MaxDeps = 128, std::size_t MaxRegions = 1>
class frame {
public:
  class dependency_group_builder {
  public:
    bool push(task_ref task) { return owner_->append_dependency(ref_, task); }

    dependency_group_ref ref() const { return ref_; }

  private:
    friend class frame;

    dependency_group_builder(frame *owner, dependency_group_ref ref)
        : owner_(owner), ref_(ref) {}

    frame *owner_ = nullptr;
    dependency_group_ref ref_;
  };

  event_tensor_ref event_tensor(attr_set attributes = {}) {
    if (error_.code != status_code::ok) {
      return invalid_event_tensor_ref();
    }
    if (sealed_) {
      error_ = {status_code::invalid_argument, 10,
                "megacu host-orch frame is sealed"};
      return invalid_event_tensor_ref();
    }
    if (event_count_ >= MaxEvents) {
      error_ = {status_code::invalid_argument, 11,
                "megacu host-orch event tensor capacity exceeded"};
      return invalid_event_tensor_ref();
    }

    auto ref = event_tensor_ref{static_cast<std::uint32_t>(event_count_)};
    events_[event_count_] = {.ref = ref, .attributes = attributes};
    ++event_count_;
    return ref;
  }

  task_ref submit(operator_slot slot, attr_set attributes = {}) {
    return publish_task(task_kind::operator_body, slot, attributes);
  }

  task_ref sync(attr_set attributes = {}) {
    return publish_task(task_kind::sync_only, invalid_operator_slot(),
                        attributes);
  }

  dependency_group_builder dependency_group() {
    return dependency_group_builder{
        this, {.first = static_cast<std::uint32_t>(dep_count_), .count = 0}};
  }

  status seal(std::uint32_t epoch = 0) {
    if (error_.code != status_code::ok) {
      return error_;
    }
    if (sealed_) {
      return {};
    }
    if constexpr (MaxRegions == 0) {
      error_ = {status_code::invalid_argument, 12,
                "megacu host-orch region capacity exceeded"};
      return error_;
    } else {
      regions_[0] = {.epoch = epoch,
                     .first_task = 0,
                     .task_count = static_cast<std::uint32_t>(task_count_),
                     .first_event = 0,
                     .event_count = static_cast<std::uint32_t>(event_count_),
                     .sealed = 1};
      region_count_ = 1;
      sealed_ = true;
      return {};
    }
  }

  status current_status() const { return error_; }

  bool sealed() const { return sealed_; }

  task_arena_view arena() {
    return {.tasks = tasks_.data(),
            .events = events_.data(),
            .deps = deps_.data(),
            .regions = regions_.data(),
            .task_count = static_cast<std::uint32_t>(task_count_),
            .event_count = static_cast<std::uint32_t>(event_count_),
            .dep_count = static_cast<std::uint32_t>(dep_count_),
            .region_count = static_cast<std::uint32_t>(region_count_),
            .task_capacity = static_cast<std::uint32_t>(MaxTasks),
            .event_capacity = static_cast<std::uint32_t>(MaxEvents),
            .dep_capacity = static_cast<std::uint32_t>(MaxDeps),
            .region_capacity = static_cast<std::uint32_t>(MaxRegions)};
  }

  std::span<device_task_record const> tasks() const {
    return {tasks_.data(), task_count_};
  }

  std::span<device_event_tensor_record const> event_tensors() const {
    return {events_.data(), event_count_};
  }

  std::span<task_ref const> deps() const { return {deps_.data(), dep_count_}; }

  std::span<arena_region const> regions() const {
    return {regions_.data(), region_count_};
  }

private:
  static constexpr auto max_record_dep_index =
      static_cast<std::uint32_t>(~std::uint16_t{0});

  bool append_dependency(dependency_group_ref &group, task_ref task) {
    if (error_.code != status_code::ok) {
      return false;
    }
    if (sealed_) {
      error_ = {status_code::invalid_argument, 10,
                "megacu host-orch frame is sealed"};
      return false;
    }
    if (dep_count_ != static_cast<std::size_t>(group.first + group.count)) {
      error_ = {status_code::invalid_argument, 17,
                "megacu host-orch dependency group is not contiguous"};
      return false;
    }
    if (dep_count_ > max_record_dep_index ||
        group.count >= max_record_dep_index) {
      error_ = {status_code::invalid_argument, 18,
                "megacu host-orch dependency range exceeds record capacity"};
      return false;
    }
    if (dep_count_ >= MaxDeps) {
      error_ = {status_code::invalid_argument, 14,
                "megacu host-orch dependency capacity exceeded"};
      return false;
    }
    deps_[dep_count_++] = task;
    ++group.count;
    return true;
  }

  task_ref publish_task(task_kind kind, operator_slot slot,
                        attr_set attributes) {
    if (error_.code != status_code::ok) {
      return invalid_task_ref();
    }
    if (sealed_) {
      error_ = {status_code::invalid_argument, 10,
                "megacu host-orch frame is sealed"};
      return invalid_task_ref();
    }
    if (task_count_ >= MaxTasks) {
      error_ = {status_code::invalid_argument, 13,
                "megacu host-orch task capacity exceeded"};
      return invalid_task_ref();
    }

    auto first_dep = static_cast<std::uint32_t>(dep_count_);
    std::uint16_t dependency_group_count = 0;
    std::uint16_t task_dep_count = 0;
    std::uint16_t inline_dep_count = 0;
    for (auto item : attributes.entries()) {
      if (item.kind == attr_kind::dependency_group) {
        ++dependency_group_count;
        first_dep = static_cast<std::uint32_t>(item.first);
        task_dep_count = static_cast<std::uint16_t>(item.second);
        if (item.first < 0 || item.second < 0 ||
            item.first > max_record_dep_index ||
            item.second > max_record_dep_index) {
          error_ = {status_code::invalid_argument, 18,
                    "megacu host-orch dependency range exceeds record "
                    "capacity"};
          return invalid_task_ref();
        }
        continue;
      }
      if (item.kind != attr_kind::dependency) {
        continue;
      }
      ++inline_dep_count;
    }

    if (dependency_group_count > 1) {
      error_ = {status_code::invalid_argument, 15,
                "megacu host-orch task has multiple dependency groups"};
      return invalid_task_ref();
    }
    if (dependency_group_count == 1 && inline_dep_count > 0) {
      error_ = {status_code::invalid_argument, 16,
                "megacu host-orch task mixes dependency group and inline "
                "dependencies"};
      return invalid_task_ref();
    }

    if (dependency_group_count == 0) {
      for (auto item : attributes.entries()) {
        if (item.kind != attr_kind::dependency) {
          continue;
        }
        if (dep_count_ > max_record_dep_index ||
            task_dep_count >= max_record_dep_index) {
          error_ = {status_code::invalid_argument, 18,
                    "megacu host-orch dependency range exceeds record "
                    "capacity"};
          return invalid_task_ref();
        }
        if (dep_count_ >= MaxDeps) {
          error_ = {status_code::invalid_argument, 14,
                    "megacu host-orch dependency capacity exceeded"};
          return invalid_task_ref();
        }
        deps_[dep_count_++] = item.task;
        ++task_dep_count;
      }
    }

    auto ref = task_ref{static_cast<std::uint32_t>(task_count_)};
    tasks_[task_count_] = {.kind = kind,
                           .op_slot = slot,
                           .dep_count = task_dep_count,
                           .first_dep = static_cast<std::uint16_t>(first_dep),
                           .attributes = attributes};
    ++task_count_;
    return ref;
  }

  std::array<device_task_record, MaxTasks> tasks_{};
  std::array<device_event_tensor_record, MaxEvents> events_{};
  std::array<task_ref, MaxDeps> deps_{};
  std::array<arena_region, MaxRegions> regions_{};
  std::size_t task_count_ = 0;
  std::size_t event_count_ = 0;
  std::size_t dep_count_ = 0;
  std::size_t region_count_ = 0;
  bool sealed_ = false;
  status error_;
};

} // namespace megacu::runtime::execution::host_orch

#undef MEGACU_HOST_ORCH_HOST_DEVICE
