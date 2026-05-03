#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace megacu {

template <class T, std::size_t Extent = std::dynamic_extent>
using span = std::span<T, Extent>;

enum class dtype : std::uint16_t {
  unknown = 0,
  f16,
  bf16,
  f32
};

enum class status_code : std::uint8_t {
  ok,
  invalid_argument,
  unsupported,
  backend_error,
  launch_error
};

struct status {
  status_code code = status_code::ok;
  std::uint16_t detail = 0;
  char const *message = nullptr;
};

struct backend_id {
  std::uint16_t value = 0;
};

struct session_id {
  std::uint64_t value = 0;
};

struct symmetric_buffer_view {
  void *data = nullptr;
  std::int64_t bytes = 0;
  backend_id backend;
  session_id session;
};

struct event_storage_view {
  symmetric_buffer_view buffer;
};

struct tensor_view {
  void *data = nullptr;
  std::int64_t bytes = 0;
  dtype type = dtype::unknown;
};

struct symmetric_tensor_view {
  symmetric_buffer_view buffer;
  dtype type = dtype::unknown;
};

}  // namespace megacu
