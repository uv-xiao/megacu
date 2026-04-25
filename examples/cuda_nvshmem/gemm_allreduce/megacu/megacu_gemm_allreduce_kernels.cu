#include "examples/cuda_nvshmem/gemm_allreduce/gemm_allreduce.h"
#include "examples/cuda_nvshmem/gemm_allreduce/golden/golden_gemm_allreduce.h"

namespace {

megacu::status to_megacu(golden_status status) {
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

golden_gemm_ar_workspace make_workspace(
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

golden_gemm_ar_launch make_launch(megacu::cuda::launch_view launch) {
  return {
      .stream = launch.stream,
      .device_ordinal = launch.device_ordinal};
}

golden_gemm_ar_problem make_problem(gemm_ar_problem problem) {
  return {
      .m = problem.m,
      .n = problem.n,
      .k = problem.k,
      .tile_m = problem.tile_m,
      .tile_n = problem.tile_n,
      .compute_ctas = 2,
      .comm_ctas = 1};
}

golden_gemm_ar_team make_team(
    megacu::nvshmem::team_view team) {
  return {
      .team = team.team,
      .team_my_pe = team.team_my_pe,
      .team_n_pes = team.team_n_pes,
      .world_my_pe = team.world_my_pe,
      .world_n_pes = team.world_n_pes,
      .cuda_device_ordinal = team.cuda_device_ordinal};
}

}  // namespace

extern "C" megacu::status megacu_cuda_gemm_allreduce_phased_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  if (team.team_n_pes <= 1) {
    return to_megacu(golden_phased_single_card_gemm_allreduce_f32(
        make_workspace(workspace, events),
        make_launch(launch),
        make_problem(problem)));
  }

  return to_megacu(golden_phased_multi_card_gemm_allreduce_f32(
      make_workspace(workspace, events),
      make_launch(launch),
      make_team(team),
      make_problem(problem)));
}

extern "C" megacu::status megacu_cuda_gemm_allreduce_overlap_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  if (team.team_n_pes <= 1) {
    return to_megacu(golden_overlap_single_card_gemm_allreduce_f32(
        make_workspace(workspace, events),
        make_launch(launch),
        make_problem(problem)));
  }

  return to_megacu(golden_overlap_multi_card_gemm_allreduce_f32(
      make_workspace(workspace, events),
      make_launch(launch),
      make_team(team),
      make_problem(problem)));
}
