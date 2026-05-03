#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include <megacu/backends/nvshmem.h>
#include <megacu/platform/cuda.h>
#include <megacu/views.h>

namespace megacu::runtime {

enum class progress_model : std::uint8_t { asap, co_resident_persistent };

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
  event_tensor_shape,
  event_tensor_storage,
  event_tensor_scope,
  event_publish,
  event_acquire,
  event_join
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

  bool push(attr item) {
    if (size_ >= entries_.size()) {
      return false;
    }
    entries_[size_++] = item;
    return true;
  }

  std::span<attr const> entries() const { return {entries_.data(), size_}; }

  std::size_t size() const { return size_; }

private:
  std::array<attr, max_attrs> entries_{};
  std::size_t size_ = 0;
};

namespace scheduler {

inline attr depends_on(task_ref task) {
  return {.kind = attr_kind::dependency, .task = task};
}

} // namespace scheduler

namespace events {

inline attr shape(std::int64_t first, std::int64_t second) {
  return {
      .kind = attr_kind::event_tensor_shape, .first = first, .second = second};
}

inline attr publish(event_tensor_ref event) {
  return {.kind = attr_kind::event_publish, .event = event};
}

inline attr acquire(event_tensor_ref event) {
  return {.kind = attr_kind::event_acquire, .event = event};
}

inline attr join(event_tensor_ref event) {
  return {.kind = attr_kind::event_join, .event = event};
}

} // namespace events

namespace dispatcher {

inline attr tile_grid(std::int64_t m_tiles, std::int64_t n_tiles) {
  return {.kind = attr_kind::dispatch_tile_grid,
          .first = m_tiles,
          .second = n_tiles};
}

} // namespace dispatcher

template <class... Attrs> attr_set attrs(Attrs... items) {
  static_assert(sizeof...(Attrs) <= attr_set::max_attrs,
                "megacu attr_set capacity exceeded");
  attr_set set;
  (set.push(items), ...);
  return set;
}

template <class Signature> struct operator_ref;

template <class Return, class... Args> struct operator_ref<Return(Args...)> {
  using function_type = Return(Args...);
  using return_type = Return;

  static constexpr std::size_t arity = sizeof...(Args);

  char const *name = nullptr;
  function_type *entry = nullptr;
};

template <class Return, class... Args>
operator_ref<Return(Args...)> op(char const *name, Return (*entry)(Args...)) {
  return {.name = name, .entry = entry};
}

template <class Signature> operator_ref<Signature> op(char const *name) {
  return {.name = name, .entry = nullptr};
}

enum class task_kind : std::uint8_t { operator_body, sync_only };

struct linked_task {
  task_kind kind = task_kind::operator_body;
  char const *op_name = nullptr;
  std::uint16_t raw_arg_count = 0;
  attr_set attributes;
  std::function<status()> invoke;
};

struct linked_event_tensor {
  event_tensor_ref ref;
  attr_set attributes;
};

namespace dispatcher {

struct dispatch_state {
  std::size_t task_count = 0;
  std::size_t event_tensor_count = 0;
  std::int64_t tile_grid_attrs = 0;
  std::int64_t total_tile_count = 0;
  std::int64_t sync_task_count = 0;
  std::int64_t sync_event_count = 0;
  std::int64_t event_publish_attrs = 0;
  std::int64_t event_acquire_attrs = 0;
  std::int64_t event_join_attrs = 0;
};

status map_explicit_attrs(dispatch_state &out,
                          std::span<linked_task const> tasks,
                          std::span<linked_event_tensor const> events = {});

} // namespace dispatcher

namespace scheduler {

status run_explicit_asap(std::span<linked_task const> tasks,
                         dispatcher::dispatch_state const &dispatch,
                         progress_model progress);

status validate_explicit_asap(std::span<linked_task const> tasks,
                              dispatcher::dispatch_state const &dispatch,
                              progress_model progress);

} // namespace scheduler

namespace target {

status run_phase(std::span<linked_task const> tasks,
                 std::span<linked_event_tensor const> events,
                 progress_model progress);

status validate_phase(std::span<linked_task const> tasks,
                      std::span<linked_event_tensor const> events,
                      progress_model progress);

} // namespace target

namespace detail {

template <class T>
inline constexpr bool is_attr_set_v =
    std::is_same_v<std::remove_cvref_t<T>, attr_set>;

template <class... Ts> struct last_type;

template <class T> struct last_type<T> {
  using type = T;
};

template <class T, class... Rest>
struct last_type<T, Rest...> : last_type<Rest...> {};

template <class... Submitted>
attr_set trailing_attrs_or_empty(Submitted &&...submitted) {
  if constexpr (sizeof...(Submitted) == 0) {
    return {};
  } else if constexpr (is_attr_set_v<typename last_type<Submitted...>::type>) {
    return std::get<sizeof...(Submitted) - 1>(
        std::forward_as_tuple(std::forward<Submitted>(submitted)...));
  } else {
    return {};
  }
}

template <class... Submitted> struct trailing_attr_count {
  static constexpr std::size_t value = 0;
};

template <class First, class... Rest>
struct trailing_attr_count<First, Rest...> {
  static constexpr std::size_t value =
      is_attr_set_v<typename last_type<First, Rest...>::type> ? 1 : 0;
};

template <class... Submitted>
inline constexpr std::size_t raw_arg_count_v =
    sizeof...(Submitted) - trailing_attr_count<Submitted...>::value;

template <std::size_t Count, class Tuple, std::size_t... Indices>
auto take_front_impl(Tuple &&tuple, std::index_sequence<Indices...>) {
  return std::tuple<std::decay_t<decltype(std::get<Indices>(tuple))>...>(
      std::get<Indices>(std::forward<Tuple>(tuple))...);
}

template <std::size_t Count, class Tuple> auto take_front(Tuple &&tuple) {
  return take_front_impl<Count>(std::forward<Tuple>(tuple),
                                std::make_index_sequence<Count>{});
}

template <class Return, class Function, class Tuple>
status call_as_status(Function *function, Tuple &tuple) {
  if (function == nullptr) {
    return {status_code::invalid_argument, 1, "missing operator entry"};
  }

  if constexpr (std::is_same_v<Return, status>) {
    return std::apply(function, tuple);
  } else if constexpr (std::is_same_v<Return, void>) {
    std::apply(function, tuple);
    return {};
  } else {
    (void)std::apply(function, tuple);
    return {};
  }
}

} // namespace detail

template <class Driver, std::size_t MaxTasks = 64, std::size_t MaxEvents = 16>
class phase {
public:
  phase(Driver &driver, progress_model progress)
      : driver_(&driver), progress_(progress) {}

  event_tensor_ref event_tensor(attr_set attributes = {}) {
    if (error_.code != status_code::ok || event_count_ >= events_.size()) {
      error_ = {status_code::invalid_argument, 5,
                "megacu event tensor capacity exceeded"};
      return invalid_event_tensor_ref();
    }

    auto id = event_tensor_ref{static_cast<std::uint32_t>(event_count_)};
    events_[event_count_] = {.ref = id, .attributes = attributes};
    ++event_count_;
    return id;
  }

  template <class Signature, class... Submitted>
  task_ref submit(operator_ref<Signature> op, Submitted &&...submitted) {
    if (error_.code != status_code::ok || task_count_ >= tasks_.size()) {
      error_ = {status_code::invalid_argument, 4,
                "megacu phase task capacity exceeded"};
      return invalid_task_ref();
    }

    static_assert(
        detail::raw_arg_count_v<Submitted...> == operator_ref<Signature>::arity,
        "megacu task args must match the native operator signature; pass "
        "optional runtime attrs as the final argument");

    auto submitted_tuple =
        std::forward_as_tuple(std::forward<Submitted>(submitted)...);
    auto raw_args = detail::take_front<detail::raw_arg_count_v<Submitted...>>(
        submitted_tuple);

    auto id = task_ref{static_cast<std::uint32_t>(task_count_)};
    tasks_[task_count_] = {
        .kind = task_kind::operator_body,
        .op_name = op.name,
        .raw_arg_count =
            static_cast<std::uint16_t>(operator_ref<Signature>::arity),
        .attributes = detail::trailing_attrs_or_empty(
            std::forward<Submitted>(submitted)...),
        .invoke = [entry = op.entry,
                   raw_args = std::move(raw_args)]() mutable -> status {
          using return_type = typename operator_ref<Signature>::return_type;
          return detail::call_as_status<return_type>(entry, raw_args);
        }};
    ++task_count_;
    return id;
  }

  task_ref sync(attr_set attributes = {}) {
    if (error_.code != status_code::ok || task_count_ >= tasks_.size()) {
      error_ = {status_code::invalid_argument, 4,
                "megacu phase task capacity exceeded"};
      return invalid_task_ref();
    }

    auto id = task_ref{static_cast<std::uint32_t>(task_count_)};
    tasks_[task_count_] = {.kind = task_kind::sync_only,
                           .op_name = "sync",
                           .raw_arg_count = 0,
                           .attributes = attributes,
                           .invoke = {}};
    ++task_count_;
    return id;
  }

  status run() {
    if (error_.code != status_code::ok) {
      return error_;
    }
    return target::run_phase(tasks(), event_tensors(), progress_);
  }

  template <class Signature, class... Submitted>
  status run(operator_ref<Signature> launcher, Submitted &&...submitted) {
    if (error_.code != status_code::ok) {
      return error_;
    }

    static_assert(
        detail::raw_arg_count_v<Submitted...> == operator_ref<Signature>::arity,
        "megacu launcher args must match the native launcher signature");

    auto validation =
        target::validate_phase(tasks(), event_tensors(), progress_);
    if (validation.code != status_code::ok) {
      return validation;
    }

    auto submitted_tuple =
        std::forward_as_tuple(std::forward<Submitted>(submitted)...);
    auto raw_args = detail::take_front<detail::raw_arg_count_v<Submitted...>>(
        submitted_tuple);
    using return_type = typename operator_ref<Signature>::return_type;
    return detail::call_as_status<return_type>(launcher.entry, raw_args);
  }

  Driver &driver() const { return *driver_; }

  progress_model progress() const { return progress_; }

  std::span<linked_task const> tasks() const {
    return {tasks_.data(), task_count_};
  }

  std::span<linked_event_tensor const> event_tensors() const {
    return {events_.data(), event_count_};
  }

private:
  Driver *driver_ = nullptr;
  progress_model progress_ = progress_model::asap;
  std::array<linked_task, MaxTasks> tasks_{};
  std::array<linked_event_tensor, MaxEvents> events_{};
  std::size_t task_count_ = 0;
  std::size_t event_count_ = 0;
  status error_;
};

template <class Driver>
phase<Driver> make_phase(Driver &driver, progress_model progress) {
  return phase<Driver>(driver, progress);
}

} // namespace megacu::runtime

namespace megacu::cuda_nvshmem {

struct driver_view {
  megacu::cuda::launch_view launch;
  megacu::nvshmem::team_view team;
};

namespace events {

inline megacu::runtime::attr symmetric_i32_storage(void *storage) {
  return {.kind = megacu::runtime::attr_kind::event_tensor_storage,
          .pointer = storage};
}

inline megacu::runtime::attr team_scope() {
  return {.kind = megacu::runtime::attr_kind::event_tensor_scope};
}

} // namespace events

} // namespace megacu::cuda_nvshmem
