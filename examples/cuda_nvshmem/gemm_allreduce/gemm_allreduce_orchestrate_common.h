#pragma once

#include "examples/cuda_nvshmem/gemm_allreduce/gemm_allreduce.h"

#include <cstdint>

#include <megacu/detail/runtime_validation.h>
#include <megacu/detail/target_metadata.h>

#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
extern "C" megacu::status megacu_cuda_gemm_allreduce_phased_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);

extern "C" megacu::status megacu_cuda_gemm_allreduce_overlap_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
#endif

namespace gemm_ar_detail {

constexpr std::int64_t kRequiredEventBytes = 16;

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

  auto team_status = megacu::detail::validate_nvshmem_team(team, 2);
  if (team_status.code != megacu::status_code::ok) {
    return team_status;
  }

  auto launch_status = megacu::detail::validate_cuda_launch(launch, team);
  if (launch_status.code != megacu::status_code::ok) {
    return launch_status;
  }

  if (workspace.a.data == nullptr || workspace.b.data == nullptr ||
      workspace.c.data == nullptr || workspace.partial.buffer.data == nullptr) {
    return {megacu::status_code::invalid_argument, 4, "missing workspace data"};
  }

  auto storage_status = megacu::detail::validate_nvshmem_symmetric_storage(
      events,
      workspace.partial,
      team,
      kRequiredEventBytes);
  if (storage_status.code != megacu::status_code::ok) {
    return storage_status;
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

  auto validation = validate_numeric_f32(workspace, problem);
  if (validation.code != megacu::status_code::ok) {
    return validation;
  }

#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
  if (progress == megacu::detail::progress_model::phased) {
    return megacu_cuda_gemm_allreduce_phased_f32(
        workspace, events, launch, team, problem);
  }
  return megacu_cuda_gemm_allreduce_overlap_f32(
      workspace, events, launch, team, problem);
#else
  return {
      megacu::status_code::unsupported,
      9,
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
