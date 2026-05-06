#include <cassert>

#include <cuda_runtime.h>

#include <megacu/runtime/device_task_arena.cuh>

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh"

namespace {

__global__ void build_recipe_on_device(megacu::runtime::task_arena_view arena,
                                       megacu::status *device_status) {
  megacu::runtime::device_orch orch{arena};
  auto args = gemm_ar::runtime_args{
      .driver = {.team = {.team_n_pes = 1}},
      .problem = {.m = 2, .n = 3, .k = 4, .tile_m = 1, .tile_n = 2}};
  auto status = gemm_ar::build_runtime_recipe(orch, args);
  if (status.code == megacu::status_code::ok) {
    status = orch.seal();
  }
  *device_status = status;
}

void require_cuda(cudaError_t error) {
  assert(error == cudaSuccess);
}

} // namespace

int main() {
  megacu::runtime::device_task_record *tasks = nullptr;
  megacu::runtime::device_event_tensor_record *events = nullptr;
  megacu::runtime::task_ref *deps = nullptr;
  megacu::runtime::arena_region *regions = nullptr;
  megacu::status *device_status = nullptr;

  require_cuda(cudaMallocManaged(&tasks, sizeof(*tasks) * 4));
  require_cuda(cudaMallocManaged(&events, sizeof(*events) * 2));
  require_cuda(cudaMallocManaged(&deps, sizeof(*deps) * 4));
  require_cuda(cudaMallocManaged(&regions, sizeof(*regions)));
  require_cuda(cudaMallocManaged(&device_status, sizeof(*device_status)));

  *device_status = {};
  auto arena = megacu::runtime::task_arena_view{
      .tasks = tasks,
      .events = events,
      .deps = deps,
      .regions = regions,
      .task_count = 0,
      .event_count = 0,
      .dep_count = 0,
      .region_count = 0,
      .task_capacity = 4,
      .event_capacity = 2,
      .dep_capacity = 4,
      .region_capacity = 1};

  build_recipe_on_device<<<1, 1>>>(arena, device_status);
  require_cuda(cudaDeviceSynchronize());

  assert(device_status->code == megacu::status_code::ok);
  assert(regions[0].sealed == 1);
  assert(regions[0].task_count == 3);
  assert(regions[0].event_count == 1);
  assert(tasks[0].op_slot == gemm_ar::runtime_slots::gemm_tile_produce);
  assert(tasks[1].kind == megacu::runtime::task_kind::sync_only);
  assert(tasks[1].dep_count == 1);
  assert(deps[tasks[1].first_dep].value == 0);
  assert(tasks[2].op_slot == gemm_ar::runtime_slots::allreduce_tile_consume);
  assert(tasks[2].dep_count == 1);
  assert(deps[tasks[2].first_dep].value == 1);

  require_cuda(cudaFree(device_status));
  require_cuda(cudaFree(regions));
  require_cuda(cudaFree(deps));
  require_cuda(cudaFree(events));
  require_cuda(cudaFree(tasks));
  return 0;
}
