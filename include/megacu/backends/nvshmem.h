#pragma once

#include <cstdint>

#include <megacu/views.h>

#if defined(MEGACU_HAS_DEVICE_NVSHMEM)
#include <nvshmem.h>
#endif

#if defined(__CUDACC__)
#define MEGACU_NVSHMEM_HOST_DEVICE __host__ __device__
#else
#define MEGACU_NVSHMEM_HOST_DEVICE
#endif

namespace megacu::nvshmem {

enum class ownership : std::uint8_t {
  external,
  owned
};

struct team_view {
  void *team = nullptr;
  std::int32_t team_my_pe = 0;
  std::int32_t team_n_pes = 1;
  std::int32_t world_my_pe = 0;
  std::int32_t world_n_pes = 1;
  std::int32_t cuda_device_ordinal = 0;
  megacu::backend_id backend;
  megacu::session_id session;
  ownership owner = ownership::external;
};

std::int32_t team_size(team_view team) noexcept;

bool has_remote_pes(team_view team) noexcept;

namespace device {

MEGACU_NVSHMEM_HOST_DEVICE inline int remote_int(
    int const *value,
    std::int64_t index,
    int pe) {
#if defined(__CUDA_ARCH__) && defined(MEGACU_HAS_DEVICE_NVSHMEM)
  return nvshmem_int_g(value + index, pe);
#else
  (void)pe;
  return value[index];
#endif
}

MEGACU_NVSHMEM_HOST_DEVICE inline float remote_float(
    float const *value,
    std::int64_t index,
    int pe) {
#if defined(__CUDA_ARCH__) && defined(MEGACU_HAS_DEVICE_NVSHMEM)
  return nvshmem_float_g(value + index, pe);
#else
  (void)pe;
  return value[index];
#endif
}

}  // namespace device

}  // namespace megacu::nvshmem

#undef MEGACU_NVSHMEM_HOST_DEVICE
