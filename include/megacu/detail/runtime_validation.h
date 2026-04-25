#pragma once

#include <cstdint>

#include <megacu/backends/nvshmem.h>
#include <megacu/platform/cuda.h>
#include <megacu/views.h>

namespace megacu::detail {

megacu::status validate_cuda_launch(
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team);

megacu::status validate_nvshmem_team(
    megacu::nvshmem::team_view team,
    std::int32_t max_supported_team_size);

megacu::status validate_nvshmem_symmetric_storage(
    megacu::event_storage_view events,
    megacu::symmetric_tensor_view partial,
    megacu::nvshmem::team_view team,
    std::int64_t required_event_bytes);

}  // namespace megacu::detail
