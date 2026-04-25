#pragma once

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"
#include "examples/cuda_nvshmem/gemm_allreduce/golden/golden_gemm_allreduce.h"

namespace gemm_ar_native {

inline megacu::status to_megacu(golden_status status) {
  switch (status.code) {
    case golden_status_code::ok:
      return {};
    case golden_status_code::launch_error:
      return {
          megacu::status_code::launch_error,
          status.detail,
          status.message};
    case golden_status_code::backend_error:
      return {
          megacu::status_code::backend_error,
          status.detail,
          status.message};
    case golden_status_code::unsupported:
      return {
          megacu::status_code::unsupported,
          status.detail,
          status.message};
    default:
      return {
          megacu::status_code::invalid_argument,
          status.detail,
          status.message};
  }
}

inline golden_gemm_ar_workspace make_workspace(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events) {
  return {
      .a = {.data = workspace.a.data, .bytes = workspace.a.bytes},
      .b = {.data = workspace.b.data, .bytes = workspace.b.bytes},
      .partial = {
          .data = workspace.partial.buffer.data,
          .bytes = workspace.partial.buffer.bytes},
      .c = {.data = workspace.c.data, .bytes = workspace.c.bytes},
      .barriers = events.buffer.data,
      .barrier_bytes = events.buffer.bytes};
}

inline golden_gemm_ar_launch make_launch(megacu::cuda::launch_view launch) {
  return {
      .stream = launch.stream,
      .device_ordinal = launch.device_ordinal};
}

inline golden_gemm_ar_problem make_problem(gemm_ar_problem problem) {
  return {
      .m = problem.m,
      .n = problem.n,
      .k = problem.k,
      .tile_m = problem.tile_m,
      .tile_n = problem.tile_n,
      .compute_ctas = 2,
      .comm_ctas = 1};
}

inline golden_gemm_ar_team make_team(
    megacu::nvshmem::team_view team) {
  return {
      .team = team.team,
      .team_my_pe = team.team_my_pe,
      .team_n_pes = team.team_n_pes,
      .world_my_pe = team.world_my_pe,
      .world_n_pes = team.world_n_pes,
      .cuda_device_ordinal = team.cuda_device_ordinal};
}

}  // namespace gemm_ar_native
