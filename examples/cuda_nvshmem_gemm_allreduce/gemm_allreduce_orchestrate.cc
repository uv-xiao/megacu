#include "examples/cuda_nvshmem_gemm_allreduce/gemm_allreduce.h"

#include <cstdint>

namespace {

constexpr std::int64_t kRequiredEventBytes = 16;

bool same_backend_session(
    megacu::symmetric_buffer_view lhs,
    megacu::symmetric_buffer_view rhs) {
  return lhs.backend.value == rhs.backend.value &&
         lhs.session.value == rhs.session.value;
}

megacu::status validate_common(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 ||
      problem.tile_m <= 0 || problem.tile_n <= 0) {
    return {megacu::status_code::invalid_argument, 1, "invalid problem shape"};
  }
  if (team.team_n_pes != 2 || team.team_my_pe < 0 ||
      team.team_my_pe >= team.team_n_pes) {
    return {megacu::status_code::invalid_argument, 2, "invalid team envelope"};
  }
  if (launch.device_ordinal != team.cuda_device_ordinal) {
    return {megacu::status_code::invalid_argument, 3, "device mismatch"};
  }
  if (workspace.a.data == nullptr || workspace.b.data == nullptr ||
      workspace.c.data == nullptr || workspace.partial.buffer.data == nullptr) {
    return {megacu::status_code::invalid_argument, 4, "missing workspace data"};
  }
  if (events.buffer.data == nullptr || events.buffer.bytes < kRequiredEventBytes) {
    return {megacu::status_code::invalid_argument, 5, "invalid event storage"};
  }

  megacu::symmetric_buffer_view team_identity{
      .backend = team.backend,
      .session = team.session};
  if (!same_backend_session(events.buffer, team_identity) ||
      !same_backend_session(workspace.partial.buffer, team_identity)) {
    return {
        megacu::status_code::invalid_argument,
        6,
        "symmetric storage session mismatch"};
  }

  return {};
}

}  // namespace

megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  return validate_common(workspace, events, launch, team, problem);
}

megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  return validate_common(workspace, events, launch, team, problem);
}
