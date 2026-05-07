#pragma once

#include "examples/cuda_nvshmem/allgather_gemm/common/allgather_gemm.h"

#include <cstdint>

#include <megacu/runtime.h>
#include <megacu/runtime/task_arena.h>

#if defined(__CUDACC__)
#define AG_GEMM_HOST_DEVICE __host__ __device__
#else
#define AG_GEMM_HOST_DEVICE
#endif

namespace ag_gemm {

struct runtime_slots {
  static constexpr megacu::runtime::operator_slot allgather_tile_produce{1};
  static constexpr megacu::runtime::operator_slot gemm_tile_consume{2};
};

struct runtime_args {
  ag_gemm_driver driver;
  float const *a = nullptr;
  float const *b = nullptr;
  float *gathered_b = nullptr;
  float *out = nullptr;
  void *events = nullptr;
  ag_gemm_problem problem;
};

AG_GEMM_HOST_DEVICE constexpr std::int64_t ceil_div(std::int64_t value,
                                                    std::int64_t divisor) {
  return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

AG_GEMM_HOST_DEVICE constexpr std::int64_t tile_rows(ag_gemm_problem problem) {
  return ceil_div(problem.m, problem.tile_m);
}

AG_GEMM_HOST_DEVICE constexpr std::int64_t tile_cols(ag_gemm_problem problem) {
  return ceil_div(problem.n, problem.tile_n);
}

AG_GEMM_HOST_DEVICE constexpr std::int64_t tile_k_cols(
    ag_gemm_problem problem) {
  return ceil_div(problem.k, problem.tile_k);
}

template <class Orch>
AG_GEMM_HOST_DEVICE megacu::status build_runtime_recipe(Orch &orch,
                                                        runtime_args args) {
  auto const k_tiles = tile_k_cols(args.problem);
  auto const n_tiles = tile_cols(args.problem);
  auto gathered = orch.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(k_tiles, n_tiles),
      megacu::runtime::event_tensor::wait_count(args.driver.team.team_n_pes),
      megacu::cuda_nvshmem::event_tensor::symmetric_storage(args.events),
      megacu::cuda_nvshmem::event_tensor::scope::team{}));

  auto allgather = orch.submit(
      runtime_slots::allgather_tile_produce,
      megacu::runtime::attrs(
          megacu::runtime::dispatcher::tile_grid(k_tiles, n_tiles),
          megacu::runtime::event_tensor::notify(gathered)));

  auto gathered_ready = orch.sync(megacu::runtime::attrs(
      megacu::runtime::scheduler::depends_on(allgather),
      megacu::runtime::event_tensor::wait(gathered)));

  (void)orch.submit(
      runtime_slots::gemm_tile_consume,
      megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(gathered_ready),
          megacu::runtime::dispatcher::tile_grid(tile_rows(args.problem),
                                                 n_tiles)));

  return orch.current_status();
}

} // namespace ag_gemm

#undef AG_GEMM_HOST_DEVICE
