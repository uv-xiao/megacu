#include "examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_arena.cuh"

#include <cstddef>
#include <cstdint>

#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
#include <cuda_runtime.h>

extern "C" megacu::status megacu_cuda_gemm_allreduce_host_orch_f32(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem,
    megacu::runtime::task_arena_view arena);

extern "C" megacu::status megacu_cuda_gemm_allreduce_seeded_orch_f32(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem,
    megacu::runtime::task_arena_view arena, megacu::status *device_status,
    std::uint32_t *construction_status);
#endif

namespace {

#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

template <class T>
megacu::status allocate_managed(T *&ptr, std::size_t count,
                                std::uint16_t detail) {
  void *raw = nullptr;
  auto status = cuda_status(cudaMallocManaged(&raw, sizeof(T) * count), detail);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  ptr = static_cast<T *>(raw);
  return {};
}

struct managed_arena {
  megacu::runtime::task_arena_view view;

  megacu::status allocate(std::uint32_t task_capacity,
                          std::uint32_t event_capacity,
                          std::uint32_t dep_capacity,
                          std::uint32_t region_capacity) {
    view.task_capacity = task_capacity;
    view.event_capacity = event_capacity;
    view.dep_capacity = dep_capacity;
    view.region_capacity = region_capacity;

    auto status = allocate_managed(view.tasks, task_capacity, 20);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    status = allocate_managed(view.events, event_capacity, 21);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    status = allocate_managed(view.deps, dep_capacity, 22);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    status = allocate_managed(view.regions, region_capacity, 23);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    status = allocate_managed(view.task_completed, task_capacity, 24);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    return allocate_managed(view.task_remaining_work, task_capacity, 25);
  }

  void release() {
    cudaFree(view.task_remaining_work);
    cudaFree(view.task_completed);
    cudaFree(view.regions);
    cudaFree(view.deps);
    cudaFree(view.events);
    cudaFree(view.tasks);
    view = {};
  }
};

void copy_host_arena(gemm_ar::host_arena_storage const &host,
                     managed_arena &device) {
  device.view.task_count = host.task_count;
  device.view.event_count = host.event_count;
  device.view.dep_count = host.dep_count;
  device.view.region_count = host.region_count;

  for (std::uint32_t index = 0; index < device.view.task_capacity; ++index) {
    device.view.tasks[index] = host.tasks[index];
    device.view.task_completed[index] = host.completed[index];
    device.view.task_remaining_work[index] = host.remaining[index];
  }
  for (std::uint32_t index = 0; index < device.view.event_capacity; ++index) {
    device.view.events[index] = host.events[index];
  }
  for (std::uint32_t index = 0; index < device.view.dep_capacity; ++index) {
    device.view.deps[index] = host.deps[index];
  }
  for (std::uint32_t index = 0; index < device.view.region_capacity; ++index) {
    device.view.regions[index] = host.regions[index];
  }
}

void clear_seeded_arena(managed_arena &arena) {
  arena.view.task_count = 0;
  arena.view.event_count = 0;
  arena.view.dep_count = 0;
  arena.view.region_count = 0;
  for (std::uint32_t index = 0; index < arena.view.task_capacity; ++index) {
    arena.view.tasks[index] = {};
    arena.view.task_completed[index] = 0;
    arena.view.task_remaining_work[index] = 0;
  }
  for (std::uint32_t index = 0; index < arena.view.event_capacity; ++index) {
    arena.view.events[index] = {};
  }
  for (std::uint32_t index = 0; index < arena.view.dep_capacity; ++index) {
    arena.view.deps[index] = {};
  }
  for (std::uint32_t index = 0; index < arena.view.region_capacity; ++index) {
    arena.view.regions[index] = {};
  }
}

megacu::status require_stream(gemm_ar_driver driver) {
  if (megacu::cuda::has_stream(driver.launch)) {
    return {};
  }
  return {megacu::status_code::invalid_argument, 1, "CUDA stream is required"};
}

#endif

megacu::status unsupported_without_cuda() {
  return {megacu::status_code::unsupported, 9,
          "CUDA numeric path was not built"};
}

} // namespace

megacu::status cuda_nvshmem_gemm_allreduce_host_orch(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem) {
#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
  auto status = require_stream(driver);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  status = cuda_status(cudaSetDevice(driver.launch.device_ordinal), 10);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

  auto args = gemm_ar::runtime_args{.driver = driver,
                                    .a = a,
                                    .b = b,
                                    .partial = partial,
                                    .out = out,
                                    .events = events,
                                    .problem = problem};
  gemm_ar::host_arena_storage host_arena;
  status = gemm_ar::build_host_arena(host_arena, args);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

  managed_arena arena;
  status = arena.allocate(host_arena.tasks.size(), host_arena.events.size(),
                          host_arena.deps.size(), host_arena.regions.size());
  if (status.code != megacu::status_code::ok) {
    arena.release();
    return status;
  }
  copy_host_arena(host_arena, arena);

  status = megacu_cuda_gemm_allreduce_host_orch_f32(
      driver, a, b, partial, out, events, problem, arena.view);
  if (status.code == megacu::status_code::ok) {
    auto stream = static_cast<cudaStream_t>(driver.launch.stream);
    status = cuda_status(cudaStreamSynchronize(stream), 30);
  }
  arena.release();
  return status;
#else
  (void)driver;
  (void)a;
  (void)b;
  (void)partial;
  (void)out;
  (void)events;
  (void)problem;
  return unsupported_without_cuda();
#endif
}

megacu::status cuda_nvshmem_gemm_allreduce_seeded_orch(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem) {
#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
  auto status = require_stream(driver);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  status = cuda_status(cudaSetDevice(driver.launch.device_ordinal), 10);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

  managed_arena arena;
  status = arena.allocate(static_cast<std::uint32_t>(gemm_ar::kTaskCapacity),
                          static_cast<std::uint32_t>(gemm_ar::kEventCapacity),
                          static_cast<std::uint32_t>(gemm_ar::kDepCapacity),
                          static_cast<std::uint32_t>(gemm_ar::kRegionCapacity));
  if (status.code != megacu::status_code::ok) {
    arena.release();
    return status;
  }
  clear_seeded_arena(arena);

  megacu::status *device_status = nullptr;
  std::uint32_t *construction_status = nullptr;
  status = allocate_managed(device_status, 1, 40);
  if (status.code != megacu::status_code::ok) {
    arena.release();
    return status;
  }
  status = allocate_managed(construction_status, 1, 41);
  if (status.code != megacu::status_code::ok) {
    cudaFree(device_status);
    arena.release();
    return status;
  }
  *device_status = {};
  *construction_status = 0;

  status = megacu_cuda_gemm_allreduce_seeded_orch_f32(
      driver, a, b, partial, out, events, problem, arena.view, device_status,
      construction_status);
  if (status.code == megacu::status_code::ok) {
    auto stream = static_cast<cudaStream_t>(driver.launch.stream);
    status = cuda_status(cudaStreamSynchronize(stream), 42);
  }
  if (status.code == megacu::status_code::ok &&
      device_status->code != megacu::status_code::ok) {
    status = *device_status;
  }

  cudaFree(construction_status);
  cudaFree(device_status);
  arena.release();
  return status;
#else
  (void)driver;
  (void)a;
  (void)b;
  (void)partial;
  (void)out;
  (void)events;
  (void)problem;
  return unsupported_without_cuda();
#endif
}
