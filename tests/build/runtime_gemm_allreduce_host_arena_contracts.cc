#include <cassert>
#include <cstdint>

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh"
#include "examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_arena.cuh"

int main() {
  gemm_ar::host_arena_storage storage;
  auto args = gemm_ar::runtime_args{
      .driver = {.team = {.team_n_pes = 1}},
      .problem = {.m = 2, .n = 3, .k = 4, .tile_m = 1, .tile_n = 2}};

  auto status = gemm_ar::build_host_arena(storage, args);
  assert(status.code == megacu::status_code::ok);

  auto arena = storage.view();
  assert(arena.tasks != nullptr);
  assert(arena.events != nullptr);
  assert(arena.deps != nullptr);
  assert(arena.regions != nullptr);
  assert(arena.task_completed != nullptr);
  assert(arena.task_remaining_work != nullptr);
  assert(arena.task_count == 3);
  assert(arena.event_count == 1);
  assert(arena.dep_count == 2);
  assert(arena.region_count == 1);
  assert(arena.task_capacity == storage.tasks.size());
  assert(arena.event_capacity == storage.events.size());
  assert(arena.dep_capacity == storage.deps.size());
  assert(arena.region_capacity == storage.regions.size());
  assert(arena.regions[0].sealed == 1);

  std::int64_t tiles =
      gemm_ar::tile_rows(args.problem) * gemm_ar::tile_cols(args.problem);
  for (std::uint32_t task = 0; task < arena.task_count; ++task) {
    assert(arena.task_completed[task] == 0);
    auto expected_remaining =
        arena.tasks[task].kind == megacu::runtime::task_kind::sync_only
            ? 1
            : static_cast<std::uint32_t>(tiles);
    assert(arena.task_remaining_work[task] == expected_remaining);
  }
  return 0;
}
