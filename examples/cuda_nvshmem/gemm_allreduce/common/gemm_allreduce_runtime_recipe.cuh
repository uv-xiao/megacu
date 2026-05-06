#pragma once

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"

#include <cstdint>

#include <megacu/runtime.h>
#include <megacu/runtime/task_arena.h>

#if defined(MEGACU_RUNTIME_HOST_DEVICE)
#define GEMM_AR_HOST_DEVICE MEGACU_RUNTIME_HOST_DEVICE
#elif defined(__CUDACC__)
#define GEMM_AR_HOST_DEVICE __host__ __device__
#else
#define GEMM_AR_HOST_DEVICE
#endif

namespace gemm_ar {

struct runtime_slots {
  static constexpr megacu::runtime::operator_slot gemm_tile_produce{1};
  static constexpr megacu::runtime::operator_slot allreduce_tile_consume{2};
};

struct runtime_args {
  gemm_ar_driver driver;
  float const *a = nullptr;
  float const *b = nullptr;
  float *partial = nullptr;
  float *out = nullptr;
  void *events = nullptr;
  gemm_ar_problem problem;
};

GEMM_AR_HOST_DEVICE constexpr std::int64_t ceil_div(std::int64_t value,
                                                    std::int64_t divisor) {
  return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

GEMM_AR_HOST_DEVICE constexpr std::int64_t tile_rows(gemm_ar_problem problem) {
  return ceil_div(problem.m, problem.tile_m);
}

GEMM_AR_HOST_DEVICE constexpr std::int64_t tile_cols(gemm_ar_problem problem) {
  return ceil_div(problem.n, problem.tile_n);
}

template <class Orch>
GEMM_AR_HOST_DEVICE megacu::status build_runtime_recipe(Orch &orch,
                                                        runtime_args args) {
  auto const m_tiles = tile_rows(args.problem);
  auto const n_tiles = tile_cols(args.problem);
  auto ready = orch.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(m_tiles, n_tiles),
      megacu::runtime::event_tensor::wait_count(args.driver.team.team_n_pes),
      megacu::cuda_nvshmem::event_tensor::symmetric_storage(args.events),
      megacu::cuda_nvshmem::event_tensor::scope::team{}));

  auto gemm = orch.submit(
      runtime_slots::gemm_tile_produce,
      megacu::runtime::attrs(
          megacu::runtime::dispatcher::tile_grid(m_tiles, n_tiles),
          megacu::runtime::event_tensor::notify(ready)));

  auto wait_ready = orch.sync(megacu::runtime::attrs(
      megacu::runtime::scheduler::depends_on(gemm),
      megacu::runtime::event_tensor::wait(ready)));

  (void)orch.submit(
      runtime_slots::allreduce_tile_consume,
      megacu::runtime::attrs(
          megacu::runtime::scheduler::depends_on(wait_ready),
          megacu::runtime::dispatcher::tile_grid(m_tiles, n_tiles)));

  return orch.current_status();
}

} // namespace gemm_ar

#undef GEMM_AR_HOST_DEVICE
