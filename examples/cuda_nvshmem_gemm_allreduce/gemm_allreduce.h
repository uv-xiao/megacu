#pragma once

#include <cstddef>
#include <cstdint>

#include <megacu/backends/nvshmem.h>
#include <megacu/platform/cuda.h>
#include <megacu/program.h>
#include <megacu/views.h>

struct m_tiles_extent;
struct n_tiles_extent;
struct output_tile_domain;
struct rank_domain;
struct compute_lane;
struct reduce_lane;
struct partial_ready_event;
struct gemm_ar_workspace_slot;
struct event_storage_slot;

namespace ops {
struct gemm_tile_produce {
  static constexpr auto name = "gemm_tile_produce";
};

struct allreduce_tile_consume {
  static constexpr auto name = "allreduce_tile_consume";
};
}  // namespace ops

struct gemm_ar_workspace {
  megacu::tensor_view a;
  megacu::tensor_view b;
  megacu::symmetric_tensor_view partial;
  megacu::tensor_view c;
  megacu::span<std::byte> scratch;
};

struct gemm_ar_problem {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::int32_t tile_m = 0;
  std::int32_t tile_n = 0;
};

struct gemm_allreduce_phased_program {
  static void describe(megacu::program_builder &p) {
    auto m_tiles = p.extent<m_tiles_extent>("m_tiles");
    auto n_tiles = p.extent<n_tiles_extent>("n_tiles");
    auto tile = p.domain<output_tile_domain>("output_tile", m_tiles, n_tiles);
    auto rank = p.domain<rank_domain>("rank", p.backend_extent("team_size"));

    auto compute = p.participant<compute_lane>("compute");
    auto reduce = p.participant<reduce_lane>("reduce");

    auto ws = p.resource<gemm_ar_workspace_slot, gemm_ar_workspace>("workspace");
    auto events =
        p.resource<event_storage_slot, megacu::event_storage_view>("events");

    auto partial_ready = p.event<partial_ready_event>(
        "partial_ready",
        megacu::over(tile, rank),
        megacu::remote_event{
            .from = compute,
            .to = reduce,
            .storage = events,
            .scope = megacu::memory_scope::remote_team});

    p.submit(
        ops::gemm_tile_produce{},
        megacu::over(tile),
        megacu::place(compute),
        megacu::args()
            .a(ws.a)
            .b(ws.b)
            .partial(ws.partial)
            .release(partial_ready.release()));

    p.submit(
        ops::allreduce_tile_consume{},
        megacu::over(tile),
        megacu::place(reduce),
        megacu::args()
            .partial(ws.partial)
            .out(ws.c)
            .acquire(partial_ready.acquire_all(rank)));
  }
};

struct gemm_allreduce_overlap_program {
  static void describe(megacu::program_builder &p) {
    auto m_tiles = p.extent<m_tiles_extent>("m_tiles");
    auto n_tiles = p.extent<n_tiles_extent>("n_tiles");
    auto tile = p.domain<output_tile_domain>("output_tile", m_tiles, n_tiles);
    auto rank = p.domain<rank_domain>("rank", p.backend_extent("team_size"));

    auto compute = p.participant<compute_lane>("compute");
    auto reduce = p.participant<reduce_lane>("reduce");

    auto ws = p.resource<gemm_ar_workspace_slot, gemm_ar_workspace>("workspace");
    auto events =
        p.resource<event_storage_slot, megacu::event_storage_view>("events");

    auto partial_ready = p.event<partial_ready_event>(
        "partial_ready",
        megacu::over(tile, rank),
        megacu::remote_event{
            .from = compute,
            .to = reduce,
            .storage = events,
            .scope = megacu::memory_scope::remote_team});

    p.submit(
        ops::gemm_tile_produce{},
        megacu::over(tile),
        megacu::place(compute),
        megacu::args()
            .a(ws.a)
            .b(ws.b)
            .partial(ws.partial)
            .release(partial_ready.release()));

    p.submit(
        ops::allreduce_tile_consume{},
        megacu::over(tile),
        megacu::place(reduce),
        megacu::args()
            .partial(ws.partial)
            .out(ws.c)
            .acquire(partial_ready.acquire_all(
                rank,
                megacu::event_wait::blocking_device())));
  }
};

megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);

megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
