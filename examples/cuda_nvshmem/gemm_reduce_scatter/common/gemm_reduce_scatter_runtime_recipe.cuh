#pragma once

#include "examples/cuda_nvshmem/gemm_reduce_scatter/common/gemm_reduce_scatter.h"

#include <cstdint>

#include <megacu/runtime.h>
#include <megacu/runtime/task_arena.h>

#if defined(__CUDACC__)
#define GEMM_RS_HOST_DEVICE __host__ __device__
#else
#define GEMM_RS_HOST_DEVICE
#endif

namespace gemm_rs {

struct runtime_slots {
  static constexpr megacu::runtime::operator_slot gemm_tile_produce{1};
  static constexpr megacu::runtime::operator_slot reduce_scatter_tile_consume{
      2};
};

struct runtime_args {
  gemm_rs_driver driver;
  float const *a = nullptr;
  float const *b = nullptr;
  float *partial = nullptr;
  float *out = nullptr;
  void *events = nullptr;
  gemm_rs_problem problem;
};

GEMM_RS_HOST_DEVICE constexpr std::int64_t ceil_div(std::int64_t value,
                                                    std::int64_t divisor) {
  return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

GEMM_RS_HOST_DEVICE constexpr std::int64_t tile_rows(gemm_rs_problem problem) {
  return ceil_div(problem.m, problem.tile_m);
}

GEMM_RS_HOST_DEVICE constexpr std::int64_t tile_cols(gemm_rs_problem problem) {
  return ceil_div(problem.n, problem.tile_n);
}

template <class Orch>
GEMM_RS_HOST_DEVICE megacu::status build_runtime_recipe(Orch &orch,
                                                        runtime_args args) {
  auto const m_tiles = tile_rows(args.problem);
  auto const n_tiles = tile_cols(args.problem);
  auto ready = orch.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(m_tiles, n_tiles),
      megacu::runtime::event_tensor::wait_count(args.driver.team.team_n_pes),
      megacu::cuda_nvshmem::event_tensor::symmetric_storage(args.events),
      megacu::cuda_nvshmem::event_tensor::scope::team{}));
  auto const tiles = m_tiles * n_tiles;

  for (std::int64_t tile = 0; tile < tiles; ++tile) {
    auto gemm = orch.submit(
        runtime_slots::gemm_tile_produce,
        megacu::runtime::attrs(
            megacu::runtime::dispatcher::single_tile(tile),
            megacu::runtime::event_tensor::notify(ready)));

    auto wait_ready = orch.sync(megacu::runtime::attrs(
        megacu::runtime::scheduler::depends_on(gemm),
        megacu::runtime::event_tensor::wait(ready, tile)));

    (void)orch.submit(
        runtime_slots::reduce_scatter_tile_consume,
        megacu::runtime::attrs(
            megacu::runtime::scheduler::depends_on(wait_ready),
            megacu::runtime::dispatcher::single_tile(tile)));
  }

  return orch.current_status();
}

} // namespace gemm_rs

#undef GEMM_RS_HOST_DEVICE
