#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <megacu/backends/nvshmem.h>
#include <megacu/platform/cuda.h>
#include <megacu/views.h>

#if defined(__CUDACC__)
#define MEGACU_RUNTIME_HOST_DEVICE __host__ __device__
#else
#define MEGACU_RUNTIME_HOST_DEVICE
#endif

namespace megacu::runtime {

enum class progress_model : std::uint8_t { asap, device_persistent };

struct task_ref {
  std::uint32_t value = 0;
};

struct event_tensor_ref {
  std::uint32_t value = 0;
};

constexpr task_ref invalid_task_ref() {
  return {static_cast<std::uint32_t>(~std::uint32_t{0})};
}

constexpr event_tensor_ref invalid_event_tensor_ref() {
  return {static_cast<std::uint32_t>(~std::uint32_t{0})};
}

constexpr bool operator==(task_ref lhs, task_ref rhs) {
  return lhs.value == rhs.value;
}

constexpr bool operator==(event_tensor_ref lhs, event_tensor_ref rhs) {
  return lhs.value == rhs.value;
}

enum class attr_kind : std::uint8_t {
  dependency,
  dispatch_tile_grid,
  dispatch_single_tile,
  event_tensor_shape,
  event_tensor_wait_count,
  event_tensor_storage,
  event_tensor_scope,
  event_notify,
  event_wait,
  event_trigger
};

struct attr {
  attr_kind kind = attr_kind::dependency;
  task_ref task;
  event_tensor_ref event;
  void *pointer = nullptr;
  std::int64_t first = 0;
  std::int64_t second = 0;
};

class attr_set {
public:
  static constexpr std::size_t max_attrs = 8;

  MEGACU_RUNTIME_HOST_DEVICE bool push(attr item) {
    if (size_ >= entries_.size()) {
      return false;
    }
    entries_[size_++] = item;
    return true;
  }

  MEGACU_RUNTIME_HOST_DEVICE std::span<attr const> entries() const {
    return {entries_.data(), size_};
  }

  MEGACU_RUNTIME_HOST_DEVICE std::size_t size() const { return size_; }

private:
  std::array<attr, max_attrs> entries_{};
  std::size_t size_ = 0;
};

namespace scheduler {

MEGACU_RUNTIME_HOST_DEVICE inline attr depends_on(task_ref task) {
  return {.kind = attr_kind::dependency, .task = task};
}

} // namespace scheduler

namespace event_tensor {

MEGACU_RUNTIME_HOST_DEVICE inline attr shape(std::int64_t first,
                                             std::int64_t second) {
  return {
      .kind = attr_kind::event_tensor_shape, .first = first, .second = second};
}

MEGACU_RUNTIME_HOST_DEVICE inline attr wait_count(std::int64_t count) {
  return {.kind = attr_kind::event_tensor_wait_count, .first = count};
}

MEGACU_RUNTIME_HOST_DEVICE inline attr notify(event_tensor_ref event) {
  return {.kind = attr_kind::event_notify, .event = event};
}

MEGACU_RUNTIME_HOST_DEVICE inline attr wait(event_tensor_ref event) {
  return {.kind = attr_kind::event_wait, .event = event, .first = -1};
}

MEGACU_RUNTIME_HOST_DEVICE inline attr wait(event_tensor_ref event,
                                           std::int64_t tile_id) {
  return {.kind = attr_kind::event_wait,
          .event = event,
          .first = tile_id,
          .second = 1};
}

MEGACU_RUNTIME_HOST_DEVICE inline attr trigger(event_tensor_ref event) {
  return {.kind = attr_kind::event_trigger, .event = event};
}

} // namespace event_tensor

namespace dispatcher {

MEGACU_RUNTIME_HOST_DEVICE inline attr tile_grid(std::int64_t m_tiles,
                                                 std::int64_t n_tiles) {
  return {.kind = attr_kind::dispatch_tile_grid,
          .first = m_tiles,
          .second = n_tiles};
}

MEGACU_RUNTIME_HOST_DEVICE inline attr single_tile(std::int64_t tile_id) {
  return {.kind = attr_kind::dispatch_single_tile, .first = tile_id};
}

} // namespace dispatcher

template <class... Attrs>
MEGACU_RUNTIME_HOST_DEVICE attr_set attrs(Attrs... items) {
  static_assert(sizeof...(Attrs) <= attr_set::max_attrs,
                "megacu attr_set capacity exceeded");
  attr_set set;
  (set.push(items), ...);
  return set;
}

enum class task_kind : std::uint8_t { operator_body, sync_only };

} // namespace megacu::runtime

namespace megacu::cuda_nvshmem {

struct driver_view {
  megacu::cuda::launch_view launch;
  megacu::nvshmem::team_view team;
};

namespace event_tensor {

MEGACU_RUNTIME_HOST_DEVICE inline megacu::runtime::attr
symmetric_storage(void *storage) {
  return {.kind = megacu::runtime::attr_kind::event_tensor_storage,
          .pointer = storage};
}

namespace scope {

struct team {
  MEGACU_RUNTIME_HOST_DEVICE constexpr operator megacu::runtime::attr() const {
    return {.kind = megacu::runtime::attr_kind::event_tensor_scope};
  }
};

} // namespace scope

MEGACU_RUNTIME_HOST_DEVICE inline megacu::runtime::attr team_scope() {
  return scope::team{};
}

} // namespace event_tensor

} // namespace megacu::cuda_nvshmem

#undef MEGACU_RUNTIME_HOST_DEVICE
