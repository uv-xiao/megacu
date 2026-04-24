#include "examples/cuda_nvshmem/gemm_allreduce/gemm_allreduce.h"
#include "examples/cuda_nvshmem/gemm_allreduce/golden/golden_gemm_allreduce.h"

namespace {

struct bridge_context {
  megacu::nvshmem::team_view team;
  gemm_ar_comm_ops const *ops = nullptr;
};

golden_status to_golden(megacu::status status) {
  switch (status.code) {
    case megacu::status_code::ok:
      return {};
    case megacu::status_code::backend_error:
      return {
          golden_status_code::backend_error,
          status.detail,
          status.message};
    case megacu::status_code::unsupported:
      return {
          golden_status_code::unsupported,
          status.detail,
          status.message};
    default:
      return {
          golden_status_code::invalid_argument,
          status.detail,
          status.message};
  }
}

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

golden_status bridge_sum_reduce_f32(
    void *team_context,
    void *dest,
    void const *src,
    std::int64_t elements,
    void *stream) {
  auto *context = static_cast<bridge_context *>(team_context);
  return to_golden(context->ops->sum_reduce_f32(
      context->team,
      dest,
      src,
      elements,
      stream));
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
    megacu::nvshmem::team_view team,
    bridge_context *context) {
  return {
      .team = context,
      .team_my_pe = team.team_my_pe,
      .team_n_pes = team.team_n_pes,
      .world_my_pe = team.world_my_pe,
      .world_n_pes = team.world_n_pes,
      .cuda_device_ordinal = team.cuda_device_ordinal};
}

golden_gemm_ar_comm_ops make_ops() {
  return {.sum_reduce_f32 = bridge_sum_reduce_f32};
}

}  // namespace

extern "C" megacu::status megacu_cuda_gemm_allreduce_phased_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem,
    gemm_ar_comm_ops const *ops) {
  if (team.team_n_pes <= 1) {
    return to_megacu(golden_phased_single_card_gemm_allreduce_f32(
        make_workspace(workspace, events),
        make_launch(launch),
        make_problem(problem)));
  }

  bridge_context context{.team = team, .ops = ops};
  auto golden_ops = make_ops();
  return to_megacu(golden_phased_multi_card_gemm_allreduce_f32(
      make_workspace(workspace, events),
      make_launch(launch),
      make_team(team, &context),
      make_problem(problem),
      &golden_ops));
}

extern "C" megacu::status megacu_cuda_gemm_allreduce_overlap_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem,
    gemm_ar_comm_ops const *ops) {
  if (team.team_n_pes <= 1) {
    return to_megacu(golden_overlap_single_card_gemm_allreduce_f32(
        make_workspace(workspace, events),
        make_launch(launch),
        make_problem(problem)));
  }

  bridge_context context{.team = team, .ops = ops};
  auto golden_ops = make_ops();
  return to_megacu(golden_overlap_multi_card_gemm_allreduce_f32(
      make_workspace(workspace, events),
      make_launch(launch),
      make_team(team, &context),
      make_problem(problem),
      &golden_ops));
}
