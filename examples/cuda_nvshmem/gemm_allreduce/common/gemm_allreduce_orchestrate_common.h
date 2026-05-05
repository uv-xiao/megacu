#pragma once

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"

#include <cstddef>
#include <cstdint>

#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
extern "C" megacu::status megacu_cuda_gemm_allreduce_phased_f32(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem);

#endif

namespace gemm_ar_detail {

using gemm_tile_produce_sig = megacu::status(gemm_ar_driver, float const *,
                                             float const *, float *,
                                             gemm_ar_problem);

using allreduce_tile_consume_sig = megacu::status(gemm_ar_driver, float *,
                                                  float *, gemm_ar_problem);

inline megacu::status orchestrate_impl(megacu::runtime::progress_model progress,
                                       gemm_ar_driver driver, float const *a,
                                       float const *b, float *partial,
                                       float *out, void *events,
                                       gemm_ar_problem problem) {
  auto phase = megacu::runtime::make_phase(driver, progress);

#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
  auto m_tiles = problem.tile_m == 0
                     ? 0
                     : (problem.m + problem.tile_m - 1) / problem.tile_m;
  auto n_tiles = problem.tile_n == 0
                     ? 0
                     : (problem.n + problem.tile_n - 1) / problem.tile_n;

  auto ready_events = phase.event_tensor(megacu::runtime::attrs(
      megacu::runtime::event_tensor::shape(m_tiles, n_tiles),
      megacu::runtime::event_tensor::wait_count(driver.team.team_n_pes),
      megacu::cuda_nvshmem::event_tensor::symmetric_storage(events),
      megacu::cuda_nvshmem::event_tensor::scope::team{}));

  auto gemm = phase.submit(
      megacu::runtime::op<gemm_tile_produce_sig>(ops::gemm_tile_produce::name),
      driver, a, b, partial, problem,
      megacu::runtime::attrs(
          megacu::runtime::dispatcher::tile_grid(m_tiles, n_tiles),
          megacu::runtime::event_tensor::notify(ready_events)));

  auto ready = phase.sync(
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(gemm),
                             megacu::runtime::event_tensor::wait(ready_events)));

  phase.submit(
      megacu::runtime::op<allreduce_tile_consume_sig>(
          ops::allreduce_tile_consume::name),
      driver, partial, out, problem,
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(ready)));

  return phase.run(megacu::runtime::op(ops::phased_megakernel::name,
                                       &megacu_cuda_gemm_allreduce_phased_f32),
                   driver, a, b, partial, out, events, problem);
#else
  (void)driver;
  (void)a;
  (void)b;
  (void)partial;
  (void)out;
  (void)events;
  (void)problem;
  return {megacu::status_code::unsupported, 9,
          "CUDA numeric path was not built"};
#endif
}

} // namespace gemm_ar_detail
