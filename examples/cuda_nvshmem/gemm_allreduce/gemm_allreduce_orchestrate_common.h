#pragma once

#include "examples/cuda_nvshmem/gemm_allreduce/gemm_allreduce.h"

#include <cstdint>

#include <megacu/detail/target_metadata.h>

#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
extern "C" megacu::status megacu_cuda_gemm_allreduce_phased_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem,
    gemm_ar_comm_ops const *ops);

extern "C" megacu::status megacu_cuda_gemm_allreduce_overlap_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem,
    gemm_ar_comm_ops const *ops);
#endif

namespace gemm_ar_detail {

constexpr std::int64_t kRequiredEventBytes = 16;

inline bool same_backend_session(
    megacu::symmetric_buffer_view lhs,
    megacu::symmetric_buffer_view rhs) {
  return lhs.backend.value == rhs.backend.value &&
         lhs.session.value == rhs.session.value;
}

inline megacu::status validate_common(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 ||
      problem.tile_m <= 0 || problem.tile_n <= 0) {
    return {megacu::status_code::invalid_argument, 1, "invalid problem shape"};
  }
  if ((team.team_n_pes != 1 && team.team_n_pes != 2) || team.team_my_pe < 0 ||
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
  if (events.buffer.data == nullptr ||
      events.buffer.bytes < kRequiredEventBytes) {
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

inline megacu::status validate_numeric_f32(
    gemm_ar_workspace workspace,
    gemm_ar_problem problem) {
  if (workspace.a.type != megacu::dtype::f32 ||
      workspace.b.type != megacu::dtype::f32 ||
      workspace.c.type != megacu::dtype::f32 ||
      workspace.partial.type != megacu::dtype::f32) {
    return {megacu::status_code::invalid_argument, 7, "numeric path requires f32"};
  }

  auto output_elements = problem.m * problem.n;
  auto output_bytes = output_elements * static_cast<std::int64_t>(sizeof(float));
  auto a_bytes = problem.m * problem.k * static_cast<std::int64_t>(sizeof(float));
  auto b_bytes = problem.k * problem.n * static_cast<std::int64_t>(sizeof(float));

  if (workspace.a.bytes < a_bytes || workspace.b.bytes < b_bytes ||
      workspace.c.bytes < output_bytes ||
      workspace.partial.buffer.bytes < 2 * output_bytes) {
    return {
        megacu::status_code::invalid_argument,
        8,
        "numeric workspace too small"};
  }

  return {};
}

inline megacu::status run_numeric(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem,
    megacu::detail::progress_model progress) {
  if (launch.stream == nullptr) {
    return {};
  }

  auto *ops = static_cast<gemm_ar_comm_ops const *>(team.team);
  if (team.team_n_pes > 1 &&
      (ops == nullptr || ops->sum_reduce_f32 == nullptr)) {
    return {
        megacu::status_code::invalid_argument,
        9,
        "missing f32 communication primitive"};
  }

  auto validation = validate_numeric_f32(workspace, problem);
  if (validation.code != megacu::status_code::ok) {
    return validation;
  }

#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
  if (progress == megacu::detail::progress_model::phased) {
    return megacu_cuda_gemm_allreduce_phased_f32(
        workspace, events, launch, team, problem, ops);
  }
  return megacu_cuda_gemm_allreduce_overlap_f32(
      workspace, events, launch, team, problem, ops);
#else
  return {
      megacu::status_code::unsupported,
      10,
      "CUDA numeric path was not built"};
#endif
}

inline megacu::status orchestrate_impl(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem,
    megacu::detail::metadata_header linked_metadata) {
  auto metadata_status = megacu::detail::validate(linked_metadata);
  if (metadata_status.code != megacu::status_code::ok) {
    return metadata_status;
  }

  auto validation = validate_common(workspace, events, launch, team, problem);
  if (validation.code != megacu::status_code::ok) {
    return validation;
  }
  return run_numeric(
      workspace,
      events,
      launch,
      team,
      problem,
      linked_metadata.progress);
}

}  // namespace gemm_ar_detail
