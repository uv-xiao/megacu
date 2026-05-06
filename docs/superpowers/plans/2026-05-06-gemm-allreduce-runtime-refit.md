# GEMM-AllReduce Runtime Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refit the CUDA+NVSHMEM GEMM-AllReduce Megacu example so it is built from operator tasks, sync-only tasks, EventTensor attrs, and the common runtime execution loop for both `host-orch` and `seeded-orch`.

**Architecture:** One shared GEMM-AllReduce recipe creates an EventTensor, submits a GEMM tile operator, inserts a sync-only wait task, then submits an AllReduce tile operator. `host-orch` builds a sealed arena on the host; `seeded-orch` builds the same arena inside the mega-kernel; both run through `runtime::device_persistent<execution_model, runtime::loop::block_tile, explicit_asap, tile_grid, attr_event_tensor_i32, operators>`. The manual mega-kernel remains only as a baseline; the Megacu example must not dispatch by hard-coded task ids or manually program EventTensor waits/signals.

**Tech Stack:** C++20, CUDA, NVSHMEM backend attrs, CMake/CTest, focused direct 1-host-1-device correctness first. Docker/MPI/Torch and 1-host-2-device execution use this refit as their next slice.

---

## File Structure

- `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h`
  - Rename the public Megacu ABI away from `phased`.
  - Add explicit host-orch and seeded-orch orchestrate entrypoints.
  - Keep raw CUDA-kernel-like args: `driver, a, b, partial, out, events, problem`.
- `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh`
  - New shared host/device recipe.
  - Defines operator slots and device-copyable runtime args.
  - Contains only orchestration topology, not math or launch policy.
- `examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_orchestrate.cc`
  - New host-side target glue for host-orch and seeded-orch.
  - Calls the shared recipe and launches the selected runtime entry.
- `examples/cuda_nvshmem/gemm_allreduce/megacu/megacu_gemm_allreduce.cu`
  - New CUDA Megacu implementation.
  - Contains operator bodies and the common runtime launchers.
  - Uses operator slots from arena records, not numeric task ids.
- `examples/cuda_nvshmem/gemm_allreduce/megacu/README.md`
  - Documents build/run commands and explains host-orch vs seeded-orch.
- `examples/cuda_nvshmem/gemm_allreduce/README.md`
  - Update the family description and remove the stale active Megacu `phased` wording.
- `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`
  - Build the new Megacu object and orchestrate target.
  - Stop building the legacy `phased/megacu` target.
  - Keep the manual mega-kernel object only as baseline.
- `tests/runtime/cuda_gemm_allreduce_correctness.cc`
  - Compare golden, manual baseline, Megacu host-orch, and Megacu seeded-orch for direct 1-host-1-device.
- Remove or stop using:
  - `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_orchestrate_common.h`
  - `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/CMakeLists.txt`
  - `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/gemm_allreduce_phased_orchestrate.cc`
  - `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/megacu_gemm_allreduce_phased.cu`

## Shared Runtime Contract

The recipe must produce this topology for every GEMM-AllReduce run:

```text
event ready = event_tensor(shape(m_tiles, n_tiles), wait_count(team_n_pes))
task gemm = submit(op_slot::gemm_tile_produce,
                   dispatch tile_grid(m_tiles, n_tiles),
                   notify ready)
task ready_wait = sync(depends_on(gemm), wait ready)
task allreduce = submit(op_slot::allreduce_tile_consume,
                        depends_on(ready_wait),
                        dispatch tile_grid(m_tiles, n_tiles))
seal epoch 0
```

The operator table must dispatch from `arena.tasks[task.value].op_slot`.

```cpp
template <class Context, class Arena, class Task, class Work>
__device__ void invoke(Context ctx, Arena arena, Task task, Work work) const {
  auto slot = arena.tasks[task.value].op_slot;
  if (slot == gemm_ar::runtime_slots::gemm_tile_produce) {
    gemm(ctx, work.tile_id);
  } else if (slot == gemm_ar::runtime_slots::allreduce_tile_consume) {
    allreduce(ctx, work.tile_id);
  }
}
```

The Megacu CUDA file must not contain:

```text
task_event_tensor_i32
notify_task
wait_task
block_tile_runtime
gemm_allreduce_program
phased_megakernel
if (task.value == 0)
if (task.value == 2)
```

## Task 1: Add The Shared GEMM-AllReduce Recipe Contract

**Files:**
- Create: `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h`
- Test: `tests/build/runtime_gemm_allreduce_recipe_contracts.cc`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`

- [ ] **Step 1: Write the host-only recipe contract**

Create `tests/build/runtime_gemm_allreduce_recipe_contracts.cc`:

```cpp
#include <cassert>

#include <megacu/runtime/execution/host_orch.h>

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh"

int main() {
  megacu::runtime::execution::host_orch::frame<4, 2, 4> frame;
  auto args = gemm_ar::runtime_args{
      .driver = {},
      .a = nullptr,
      .b = nullptr,
      .partial = nullptr,
      .out = nullptr,
      .events = nullptr,
      .problem = {.m = 2, .n = 3, .k = 4, .tile_m = 1, .tile_n = 2}};

  auto result = gemm_ar::build_runtime_recipe(frame, args);
  assert(result.code == megacu::status_code::ok);
  assert(frame.seal().code == megacu::status_code::ok);

  auto tasks = frame.tasks();
  auto events = frame.event_tensors();
  auto deps = frame.deps();
  auto regions = frame.regions();

  assert(events.size() == 1);
  assert(tasks.size() == 3);
  assert(deps.size() == 2);
  assert(regions.size() == 1);
  assert(regions[0].sealed == 1);

  assert(tasks[0].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[0].op_slot == gemm_ar::runtime_slots::gemm_tile_produce);
  assert(tasks[0].dep_count == 0);

  assert(tasks[1].kind == megacu::runtime::task_kind::sync_only);
  assert(tasks[1].op_slot == megacu::runtime::invalid_operator_slot());
  assert(tasks[1].dep_count == 1);
  assert(deps[tasks[1].first_dep].value == 0);

  assert(tasks[2].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[2].op_slot == gemm_ar::runtime_slots::allreduce_tile_consume);
  assert(tasks[2].dep_count == 1);
  assert(deps[tasks[2].first_dep].value == 1);

  bool task0_has_notify = false;
  bool task1_has_wait = false;
  bool task2_has_dispatch = false;
  for (auto attr : tasks[0].attributes.entries()) {
    task0_has_notify |= attr.kind == megacu::runtime::attr_kind::event_notify;
  }
  for (auto attr : tasks[1].attributes.entries()) {
    task1_has_wait |= attr.kind == megacu::runtime::attr_kind::event_wait;
  }
  for (auto attr : tasks[2].attributes.entries()) {
    task2_has_dispatch |=
        attr.kind == megacu::runtime::attr_kind::dispatch_tile_grid;
  }
  assert(task0_has_notify);
  assert(task1_has_wait);
  assert(task2_has_dispatch);
  return 0;
}
```

- [ ] **Step 2: Register and run the failing contract**

Add this target near the other build contracts in `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`:

```cmake
add_executable(megacu_runtime_gemm_allreduce_recipe_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_gemm_allreduce_recipe_contracts.cc)
target_link_libraries(megacu_runtime_gemm_allreduce_recipe_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_gemm_allreduce_recipe_contracts
  COMMAND megacu_runtime_gemm_allreduce_recipe_contracts)
```

Run:

```bash
cmake --build build --target megacu_runtime_gemm_allreduce_recipe_contracts
```

Expected: FAIL because `gemm_allreduce_runtime_recipe.cuh` does not exist.

- [ ] **Step 3: Implement the recipe header**

Create `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh`:

```cpp
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
```

- [ ] **Step 4: Add the public Megacu entrypoints**

Modify `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h` so the operator names and public ABI are:

```cpp
namespace ops {

struct gemm_tile_produce {
  static constexpr auto name = "gemm_tile_produce";
};

struct allreduce_tile_consume {
  static constexpr auto name = "allreduce_tile_consume";
};

}  // namespace ops

megacu::status cuda_nvshmem_gemm_allreduce_host_orch(
    gemm_ar_driver driver,
    float const *a,
    float const *b,
    float *partial,
    float *out,
    void *events,
    gemm_ar_problem problem);

megacu::status cuda_nvshmem_gemm_allreduce_seeded_orch(
    gemm_ar_driver driver,
    float const *a,
    float const *b,
    float *partial,
    float *out,
    void *events,
    gemm_ar_problem problem);
```

Remove `ops::phased_megakernel` and the declaration of `cuda_nvshmem_gemm_allreduce_phased_orchestrate`.

- [ ] **Step 5: Run the contract**

Run:

```bash
cmake --build build --target megacu_runtime_gemm_allreduce_recipe_contracts
ctest --test-dir build -R runtime_gemm_allreduce_recipe_contracts --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h \
  examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh \
  tests/build/runtime_gemm_allreduce_recipe_contracts.cc \
  examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt
git commit -m "test: add gemm allreduce runtime recipe contract"
```

## Task 2: Add Host-Orch Arena Materialization For CUDA Launch

**Files:**
- Create: `examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_arena.cuh`
- Test: `tests/build/runtime_gemm_allreduce_host_arena_contracts.cc`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`

- [ ] **Step 1: Write the host arena materialization contract**

Create `tests/build/runtime_gemm_allreduce_host_arena_contracts.cc`:

```cpp
#include <cassert>

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
  assert(arena.task_count == 3);
  assert(arena.event_count == 1);
  assert(arena.dep_count == 2);
  assert(arena.region_count == 1);
  assert(arena.regions[0].sealed == 1);

  assert(arena.task_completed[0] == 0);
  assert(arena.task_completed[1] == 0);
  assert(arena.task_completed[2] == 0);

  std::int64_t tiles =
      gemm_ar::tile_rows(args.problem) * gemm_ar::tile_cols(args.problem);
  assert(arena.task_remaining_work[0] == static_cast<std::uint32_t>(tiles));
  assert(arena.task_remaining_work[1] == 1);
  assert(arena.task_remaining_work[2] == static_cast<std::uint32_t>(tiles));
  return 0;
}
```

- [ ] **Step 2: Register and run the failing contract**

Add:

```cmake
add_executable(megacu_runtime_gemm_allreduce_host_arena_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_gemm_allreduce_host_arena_contracts.cc)
target_link_libraries(megacu_runtime_gemm_allreduce_host_arena_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_gemm_allreduce_host_arena_contracts
  COMMAND megacu_runtime_gemm_allreduce_host_arena_contracts)
```

Run:

```bash
cmake --build build --target megacu_runtime_gemm_allreduce_host_arena_contracts
```

Expected: FAIL because `gemm_allreduce_arena.cuh` does not exist.

- [ ] **Step 3: Implement the host arena storage helper**

Create `examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_arena.cuh`:

```cpp
#pragma once

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_runtime_recipe.cuh"

#include <array>
#include <cstdint>

#include <megacu/runtime/execution/host_orch.h>
#include <megacu/runtime/task_arena.h>

namespace gemm_ar {

struct host_arena_storage {
  std::array<megacu::runtime::device_task_record, 4> tasks{};
  std::array<megacu::runtime::device_event_tensor_record, 2> events{};
  std::array<megacu::runtime::task_ref, 4> deps{};
  std::array<megacu::runtime::arena_region, 1> regions{};
  std::array<std::uint32_t, 4> completed{};
  std::array<std::uint32_t, 4> remaining{};
  std::uint32_t task_count = 0;
  std::uint32_t event_count = 0;
  std::uint32_t dep_count = 0;
  std::uint32_t region_count = 0;

  megacu::runtime::task_arena_view view() {
    return {.tasks = tasks.data(),
            .events = events.data(),
            .deps = deps.data(),
            .regions = regions.data(),
            .task_completed = completed.data(),
            .task_remaining_work = remaining.data(),
            .task_count = task_count,
            .event_count = event_count,
            .dep_count = dep_count,
            .region_count = region_count,
            .task_capacity = static_cast<std::uint32_t>(tasks.size()),
            .event_capacity = static_cast<std::uint32_t>(events.size()),
            .dep_capacity = static_cast<std::uint32_t>(deps.size()),
            .region_capacity = static_cast<std::uint32_t>(regions.size())};
  }
};

inline void initialize_remaining_work(host_arena_storage &storage,
                                      runtime_args args) {
  auto const tiles = static_cast<std::uint32_t>(
      tile_rows(args.problem) * tile_cols(args.problem));
  for (std::uint32_t task = 0; task < storage.task_count; ++task) {
    storage.completed[task] = 0;
    storage.remaining[task] =
        storage.tasks[task].kind == megacu::runtime::task_kind::sync_only ? 1
                                                                          : tiles;
  }
}

inline megacu::status build_host_arena(host_arena_storage &storage,
                                       runtime_args args) {
  megacu::runtime::execution::host_orch::frame<4, 2, 4> frame;
  auto status = build_runtime_recipe(frame, args);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  status = frame.seal();
  if (status.code != megacu::status_code::ok) {
    return status;
  }

  auto tasks = frame.tasks();
  auto events = frame.event_tensors();
  auto deps = frame.deps();
  auto regions = frame.regions();
  storage.task_count = static_cast<std::uint32_t>(tasks.size());
  storage.event_count = static_cast<std::uint32_t>(events.size());
  storage.dep_count = static_cast<std::uint32_t>(deps.size());
  storage.region_count = static_cast<std::uint32_t>(regions.size());
  for (std::uint32_t index = 0; index < storage.task_count; ++index) {
    storage.tasks[index] = tasks[index];
  }
  for (std::uint32_t index = 0; index < storage.event_count; ++index) {
    storage.events[index] = events[index];
  }
  for (std::uint32_t index = 0; index < storage.dep_count; ++index) {
    storage.deps[index] = deps[index];
  }
  for (std::uint32_t index = 0; index < storage.region_count; ++index) {
    storage.regions[index] = regions[index];
  }
  initialize_remaining_work(storage, args);
  return {};
}

} // namespace gemm_ar
```

- [ ] **Step 4: Run the host arena contract**

Run:

```bash
cmake --build build --target megacu_runtime_gemm_allreduce_host_arena_contracts
ctest --test-dir build -R runtime_gemm_allreduce_host_arena_contracts --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_arena.cuh \
  tests/build/runtime_gemm_allreduce_host_arena_contracts.cc \
  examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt
git commit -m "feat: add gemm allreduce host arena materialization"
```

## Task 3: Implement The Common Megacu CUDA Runtime Launchers

**Files:**
- Create: `examples/cuda_nvshmem/gemm_allreduce/megacu/megacu_gemm_allreduce.cu`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`

- [ ] **Step 1: Create the CUDA file with operator bodies and launch declarations**

Create `examples/cuda_nvshmem/gemm_allreduce/megacu/megacu_gemm_allreduce.cu` with these components:

```cpp
#include "examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_arena.cuh"

#include <cuda_runtime.h>

#include <cstdint>

#include <megacu/backends/nvshmem/cuda_event_tensor.cuh>
#include <megacu/dispatcher/tile_grid_device.cuh>
#include <megacu/platform/cuda/megakernel.cuh>
#include <megacu/runtime/device_persistent.cuh>
#include <megacu/runtime/execution/seeded_orch.cuh>
#include <megacu/runtime/loop/block_tile.cuh>
#include <megacu/scheduler/explicit_asap_device.cuh>

#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
#include <nvshmem.h>
extern "C" void nvshmem_barrier_all();
#endif

namespace {

constexpr int kThreads = 128;
constexpr int kBlocks = 2;

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

__device__ std::int64_t device_ceil_div(std::int64_t value,
                                        std::int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

struct gemm_tile_produce_task {
  gemm_ar::runtime_args args;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    auto n_tiles = device_ceil_div(args.problem.n, args.problem.tile_n);
    auto tile_row = tile_id / n_tiles;
    auto tile_col = tile_id % n_tiles;
    auto row_begin = tile_row * args.problem.tile_m;
    auto col_begin = tile_col * args.problem.tile_n;
    auto tile_elements =
        static_cast<std::int64_t>(args.problem.tile_m) * args.problem.tile_n;

    for (auto offset = static_cast<std::int64_t>(ctx.thread_id());
         offset < tile_elements; offset += ctx.block_threads()) {
      auto local_row = offset / args.problem.tile_n;
      auto local_col = offset % args.problem.tile_n;
      auto row = row_begin + local_row;
      auto col = col_begin + local_col;
      if (row >= args.problem.m || col >= args.problem.n) {
        continue;
      }
      float value = 0.0f;
      for (std::int64_t kk = 0; kk < args.problem.k; ++kk) {
        value += args.a[row * args.problem.k + kk] *
                 args.b[kk * args.problem.n + col];
      }
      args.partial[row * args.problem.n + col] = value;
    }
  }
};

struct allreduce_tile_consume_task {
  gemm_ar::runtime_args args;

  template <class Context>
  __device__ void operator()(Context ctx, std::int64_t tile_id) const {
    auto n_tiles = device_ceil_div(args.problem.n, args.problem.tile_n);
    auto tile_row = tile_id / n_tiles;
    auto tile_col = tile_id % n_tiles;
    auto row_begin = tile_row * args.problem.tile_m;
    auto col_begin = tile_col * args.problem.tile_n;
    auto tile_elements =
        static_cast<std::int64_t>(args.problem.tile_m) * args.problem.tile_n;

    auto n_pes = megacu::nvshmem::team_size(args.driver.team);
    auto my_pe = args.driver.team.team_my_pe;
    for (auto offset = static_cast<std::int64_t>(ctx.thread_id());
         offset < tile_elements; offset += ctx.block_threads()) {
      auto local_row = offset / args.problem.tile_n;
      auto local_col = offset % args.problem.tile_n;
      auto row = row_begin + local_row;
      auto col = col_begin + local_col;
      if (row >= args.problem.m || col >= args.problem.n) {
        continue;
      }

      auto index = row * args.problem.n + col;
      float value = 0.0f;
      for (int pe = 0; pe < n_pes; ++pe) {
        if (pe == my_pe) {
          value += args.partial[index];
        } else {
#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
          value += megacu::nvshmem::device::remote_float(args.partial, index, pe);
#endif
        }
      }
      args.out[index] = value;
    }
  }
};

struct gemm_allreduce_operators {
  gemm_tile_produce_task gemm;
  allreduce_tile_consume_task allreduce;

  template <class Context, class Arena, class Task, class Work>
  __device__ void invoke(Context ctx, Arena arena, Task task, Work work) const {
    auto slot = arena.tasks[task.value].op_slot;
    if (slot == gemm_ar::runtime_slots::gemm_tile_produce) {
      gemm(ctx, work.tile_id);
    } else if (slot == gemm_ar::runtime_slots::allreduce_tile_consume) {
      allreduce(ctx, work.tile_id);
    }
  }
};

struct host_orch_execution {
  megacu::runtime::task_arena_view arena;

  template <class Context>
  __device__ megacu::runtime::task_arena_view bind(Context) const {
    return arena;
  }

  template <class Context>
  __device__ bool construct(Context, megacu::runtime::task_arena_view &) const {
    return true;
  }
};

struct seeded_gemm_ar_recipe {
  gemm_ar::runtime_args args;

  template <class Orch>
  __device__ megacu::status operator()(Orch &orch) const {
    auto status = gemm_ar::build_runtime_recipe(orch, args);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    return orch.seal();
  }
};

template <class ExecutionModel>
using gemm_ar_runtime = megacu::runtime::device_persistent<
    ExecutionModel,
    megacu::runtime::loop::block_tile,
    megacu::scheduler::device::explicit_asap,
    megacu::dispatcher::device::tile_grid,
    megacu::backend::nvshmem::cuda::attr_event_tensor_i32,
    gemm_allreduce_operators>;

template <class ExecutionModel>
megacu::status launch_runtime(gemm_ar_driver driver,
                              gemm_ar::runtime_args args,
                              megacu::runtime::task_arena_view arena,
                              ExecutionModel execution) {
  auto status = cuda_status(cudaSetDevice(driver.launch.device_ordinal), 10);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  auto stream = static_cast<cudaStream_t>(driver.launch.stream);
  auto tiles = gemm_ar::tile_rows(args.problem) * gemm_ar::tile_cols(args.problem);
  auto n_pes = megacu::nvshmem::team_size(driver.team);
  auto *barriers = static_cast<int *>(args.events);

  status = cuda_status(
      cudaMemsetAsync(barriers, 0, tiles * n_pes * sizeof(int), stream), 11);
  if (status.code != megacu::status_code::ok) {
    return status;
  }

#ifdef MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM
  if (megacu::nvshmem::has_remote_pes(driver.team)) {
    status = cuda_status(cudaStreamSynchronize(stream), 12);
    if (status.code != megacu::status_code::ok) {
      return status;
    }
    nvshmem_barrier_all();
  }
#else
  if (megacu::nvshmem::has_remote_pes(driver.team)) {
    return {megacu::status_code::unsupported, 13,
            "device-side NVSHMEM path was not built"};
  }
#endif

  auto runtime = gemm_ar_runtime<ExecutionModel>{
      .execution = execution,
      .loop = {},
      .scheduler = {},
      .dispatcher = {},
      .event_tensor = {.event_tensor = {.events = barriers,
                                        .tiles = tiles,
                                        .my_pe = driver.team.team_my_pe,
                                        .n_pes = n_pes}},
      .operators = {.gemm = {.args = args}, .allreduce = {.args = args}}};

  return cuda_status(megacu::platform::cuda::launch_megakernel(
                         stream, {.blocks = kBlocks, .threads = kThreads},
                         runtime),
                     14);
}

} // namespace
```

- [ ] **Step 2: Add host-orch and seeded-orch exported launchers**

Append exported functions to `megacu_gemm_allreduce.cu`:

```cpp
extern "C" megacu::status megacu_cuda_gemm_allreduce_host_orch_f32(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem,
    megacu::runtime::task_arena_view arena) {
  if (!megacu::cuda::has_stream(driver.launch)) {
    return {megacu::status_code::invalid_argument, 1,
            "CUDA stream is required"};
  }
  auto args = gemm_ar::runtime_args{.driver = driver,
                                    .a = a,
                                    .b = b,
                                    .partial = partial,
                                    .out = out,
                                    .events = events,
                                    .problem = problem};
  return launch_runtime(driver, args, arena, host_orch_execution{.arena = arena});
}

extern "C" megacu::status megacu_cuda_gemm_allreduce_seeded_orch_f32(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem,
    megacu::runtime::task_arena_view arena,
    megacu::status *device_status,
    std::uint32_t *construction_status) {
  if (!megacu::cuda::has_stream(driver.launch)) {
    return {megacu::status_code::invalid_argument, 1,
            "CUDA stream is required"};
  }
  auto args = gemm_ar::runtime_args{.driver = driver,
                                    .a = a,
                                    .b = b,
                                    .partial = partial,
                                    .out = out,
                                    .events = events,
                                    .problem = problem};
  auto execution =
      megacu::runtime::execution::seeded_orch::model<seeded_gemm_ar_recipe>{
          .arena = arena,
          .recipe = seeded_gemm_ar_recipe{.args = args},
          .device_status = device_status,
          .construction_status = construction_status};
  return launch_runtime(driver, args, arena, execution);
}
```

- [ ] **Step 3: Register the new CUDA object**

Replace the old Megacu object source in `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`:

```cmake
add_library(megacu_gemm_allreduce_cuda_native OBJECT
  megacu/megacu_gemm_allreduce.cu)
```

Run:

```bash
cmake --build build --target megacu_gemm_allreduce_cuda_native
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add examples/cuda_nvshmem/gemm_allreduce/megacu/megacu_gemm_allreduce.cu \
  examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt
git commit -m "feat: add gemm allreduce common runtime launchers"
```

## Task 4: Add Host-Side Orchestrate Glue For Both Execution Models

**Files:**
- Create: `examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_orchestrate.cc`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`

- [ ] **Step 1: Create host glue declarations**

Create `examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_orchestrate.cc`:

```cpp
#include "examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_arena.cuh"

#include <cuda_runtime.h>

#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
extern "C" megacu::status megacu_cuda_gemm_allreduce_host_orch_f32(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem,
    megacu::runtime::task_arena_view arena);

extern "C" megacu::status megacu_cuda_gemm_allreduce_seeded_orch_f32(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem,
    megacu::runtime::task_arena_view arena,
    megacu::status *device_status,
    std::uint32_t *construction_status);
#endif

namespace {

megacu::status cuda_status(cudaError_t error, std::uint16_t detail) {
  if (error == cudaSuccess) {
    return {};
  }
  return {megacu::status_code::launch_error, detail, cudaGetErrorString(error)};
}

template <class T>
megacu::status allocate_managed(T **ptr, std::size_t count,
                                std::uint16_t detail) {
  auto error = cudaMallocManaged(ptr, sizeof(T) * count);
  return cuda_status(error, detail);
}

megacu::status materialize_managed_arena(
    gemm_ar::host_arena_storage const &host,
    megacu::runtime::task_arena_view &arena) {
  auto status = allocate_managed(&arena.tasks, host.tasks.size(), 20);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.events, host.events.size(), 21);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.deps, host.deps.size(), 22);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.regions, host.regions.size(), 23);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.task_completed, host.completed.size(), 24);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.task_remaining_work, host.remaining.size(), 25);
  if (status.code != megacu::status_code::ok) return status;

  arena.task_count = host.task_count;
  arena.event_count = host.event_count;
  arena.dep_count = host.dep_count;
  arena.region_count = host.region_count;
  arena.task_capacity = static_cast<std::uint32_t>(host.tasks.size());
  arena.event_capacity = static_cast<std::uint32_t>(host.events.size());
  arena.dep_capacity = static_cast<std::uint32_t>(host.deps.size());
  arena.region_capacity = static_cast<std::uint32_t>(host.regions.size());

  for (std::uint32_t index = 0; index < arena.task_capacity; ++index) {
    arena.tasks[index] = host.tasks[index];
    arena.task_completed[index] = host.completed[index];
    arena.task_remaining_work[index] = host.remaining[index];
  }
  for (std::uint32_t index = 0; index < arena.event_capacity; ++index) {
    arena.events[index] = host.events[index];
  }
  for (std::uint32_t index = 0; index < arena.dep_capacity; ++index) {
    arena.deps[index] = host.deps[index];
  }
  for (std::uint32_t index = 0; index < arena.region_capacity; ++index) {
    arena.regions[index] = host.regions[index];
  }
  return {};
}

void free_managed_arena(megacu::runtime::task_arena_view arena) {
  cudaFree(arena.task_remaining_work);
  cudaFree(arena.task_completed);
  cudaFree(arena.regions);
  cudaFree(arena.deps);
  cudaFree(arena.events);
  cudaFree(arena.tasks);
}

} // namespace
```

- [ ] **Step 2: Add host-orch orchestrate**

Append:

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_host_orch(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem) {
#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
  auto args = gemm_ar::runtime_args{.driver = driver,
                                    .a = a,
                                    .b = b,
                                    .partial = partial,
                                    .out = out,
                                    .events = events,
                                    .problem = problem};
  gemm_ar::host_arena_storage host_arena;
  auto status = gemm_ar::build_host_arena(host_arena, args);
  if (status.code != megacu::status_code::ok) {
    return status;
  }
  megacu::runtime::task_arena_view device_arena;
  status = materialize_managed_arena(host_arena, device_arena);
  if (status.code != megacu::status_code::ok) {
    free_managed_arena(device_arena);
    return status;
  }
  status = megacu_cuda_gemm_allreduce_host_orch_f32(
      driver, a, b, partial, out, events, problem, device_arena);
  auto stream = static_cast<cudaStream_t>(driver.launch.stream);
  if (status.code == megacu::status_code::ok) {
    status = cuda_status(cudaStreamSynchronize(stream), 30);
  }
  free_managed_arena(device_arena);
  return status;
#else
  (void)driver; (void)a; (void)b; (void)partial; (void)out; (void)events;
  (void)problem;
  return {megacu::status_code::unsupported, 9,
          "CUDA numeric path was not built"};
#endif
}
```

- [ ] **Step 3: Add seeded-orch orchestrate**

Append:

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_seeded_orch(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem) {
#ifdef MEGACU_HAS_CUDA_NUMERIC_PATH
  megacu::runtime::task_arena_view arena;
  auto status = allocate_managed(&arena.tasks, 4, 40);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.events, 2, 41);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.deps, 4, 42);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.regions, 1, 43);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.task_completed, 4, 44);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&arena.task_remaining_work, 4, 45);
  if (status.code != megacu::status_code::ok) return status;

  arena.task_count = 0;
  arena.event_count = 0;
  arena.dep_count = 0;
  arena.region_count = 0;
  arena.task_capacity = 4;
  arena.event_capacity = 2;
  arena.dep_capacity = 4;
  arena.region_capacity = 1;
  for (std::uint32_t index = 0; index < 4; ++index) {
    arena.task_completed[index] = 0;
    arena.task_remaining_work[index] =
        index == 1 ? 1 : static_cast<std::uint32_t>(
                         gemm_ar::tile_rows(problem) * gemm_ar::tile_cols(problem));
  }
  arena.regions[0] = {};

  megacu::status *device_status = nullptr;
  std::uint32_t *construction_status = nullptr;
  status = allocate_managed(&device_status, 1, 46);
  if (status.code != megacu::status_code::ok) return status;
  status = allocate_managed(&construction_status, 1, 47);
  if (status.code != megacu::status_code::ok) return status;
  *device_status = {};
  *construction_status =
      megacu::runtime::execution::seeded_orch::construction_pending;

  status = megacu_cuda_gemm_allreduce_seeded_orch_f32(
      driver, a, b, partial, out, events, problem, arena, device_status,
      construction_status);
  auto stream = static_cast<cudaStream_t>(driver.launch.stream);
  if (status.code == megacu::status_code::ok) {
    status = cuda_status(cudaStreamSynchronize(stream), 48);
  }
  if (status.code == megacu::status_code::ok &&
      device_status->code != megacu::status_code::ok) {
    status = *device_status;
  }
  cudaFree(construction_status);
  cudaFree(device_status);
  free_managed_arena(arena);
  return status;
#else
  (void)driver; (void)a; (void)b; (void)partial; (void)out; (void)events;
  (void)problem;
  return {megacu::status_code::unsupported, 9,
          "CUDA numeric path was not built"};
#endif
}
```

During implementation, consolidate repeated allocation cleanup so failures do not leak managed allocations.

- [ ] **Step 4: Build the orchestrate target**

Change `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt` so the target is:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_megacu
  SOURCES megacu/gemm_allreduce_orchestrate.cc
  COMPONENTS cuda_nvshmem_static
  PROGRESS asap)
```

Attach `$<TARGET_OBJECTS:megacu_gemm_allreduce_cuda_native>` to `cuda_nvshmem_gemm_allreduce_megacu`.

Run:

```bash
cmake --build build --target cuda_nvshmem_gemm_allreduce_megacu
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/cuda_nvshmem/gemm_allreduce/megacu/gemm_allreduce_orchestrate.cc \
  examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt
git commit -m "feat: add gemm allreduce runtime orchestrate glue"
```

## Task 5: Update Correctness Test To Compare Golden, Baseline, Host-Orch, Seeded-Orch

**Files:**
- Modify: `tests/runtime/cuda_gemm_allreduce_correctness.cc`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`

- [ ] **Step 1: Add baseline declaration to the test**

Add this declaration after includes in `tests/runtime/cuda_gemm_allreduce_correctness.cc`:

```cpp
extern "C" megacu::status
manual_megakernel_cuda_nvshmem_gemm_allreduce_phased_f32(
    gemm_ar_driver driver, float const *a, float const *b, float *partial,
    float *out, void *events, gemm_ar_problem problem);
```

The exported baseline symbol can keep its existing name for this slice because it is a baseline-only ABI.

- [ ] **Step 2: Replace the single Megacu call with three runner calls**

Add:

```cpp
void run_megacu_case(
    char const *label,
    megacu::status (*fn)(gemm_ar_driver, float const *, float const *, float *,
                         float *, void *, gemm_ar_problem),
    gemm_ar_driver driver,
    void const *a,
    void const *b,
    void *partial,
    void *c,
    void *events,
    cudaStream_t stream,
    std::vector<float> const &host_a,
    std::vector<float> const &host_b,
    std::vector<float> &host_c) {
  reset_outputs(c, partial, events, stream);
  auto status = fn(driver, static_cast<float const *>(a),
                   static_cast<float const *>(b), static_cast<float *>(partial),
                   static_cast<float *>(c), events,
                   gemm_ar_problem{.m = kM,
                                   .n = kN,
                                   .k = kK,
                                   .tile_m = 1,
                                   .tile_n = 2});
  assert(status.code == megacu::status_code::ok);
  require_cuda(cudaMemcpyAsync(host_c.data(), c, kCBytes,
                               cudaMemcpyDeviceToHost, stream));
  require_cuda(cudaStreamSynchronize(stream));
  check_expected(host_a, host_b, host_c, 1.0f, label);
}
```

Then call:

```cpp
run_megacu_case("baseline_manual_single",
                manual_megakernel_cuda_nvshmem_gemm_allreduce_phased_f32,
                driver, a, b, partial, c, events, stream, host_a, host_b,
                host_c);
run_megacu_case("megacu_host_orch_single",
                cuda_nvshmem_gemm_allreduce_host_orch,
                driver, a, b, partial, c, events, stream, host_a, host_b,
                host_c);
run_megacu_case("megacu_seeded_orch_single",
                cuda_nvshmem_gemm_allreduce_seeded_orch,
                driver, a, b, partial, c, events, stream, host_a, host_b,
                host_c);
```

Remove the call to `cuda_nvshmem_gemm_allreduce_phased_orchestrate`.

- [ ] **Step 3: Link the test to the new target**

Update the CMake target that builds `cuda_gemm_allreduce_correctness` so it links:

```cmake
cuda_nvshmem_gemm_allreduce_megacu
manual_megakernel_cuda_nvshmem_gemm_allreduce
golden_cuda_nvshmem_gemm_allreduce
```

- [ ] **Step 4: Run the direct correctness test**

Run:

```bash
cmake --build build --target cuda_gemm_allreduce_correctness
ctest --test-dir build -R cuda_gemm_allreduce_correctness --output-on-failure
```

Expected: PASS on a CUDA-capable machine. If no CUDA device is present, document the exact failure and keep the compile target passing.

- [ ] **Step 5: Commit**

```bash
git add tests/runtime/cuda_gemm_allreduce_correctness.cc \
  examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt
git commit -m "test: compare gemm allreduce runtime variants"
```

## Task 6: Remove Legacy Megacu Proof Path And Tighten Guards

**Files:**
- Delete: `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_orchestrate_common.h`
- Delete: `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/CMakeLists.txt`
- Delete: `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/gemm_allreduce_phased_orchestrate.cc`
- Delete: `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/megacu_gemm_allreduce_phased.cu`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`

- [ ] **Step 1: Remove the legacy build edge**

Remove:

```cmake
add_subdirectory(phased/megacu)
```

Remove tests that target `phased/megacu` paths and replace them with checks against `examples/cuda_nvshmem/gemm_allreduce/megacu`.

- [ ] **Step 2: Add guards for the new architecture**

Add these tests:

```cmake
add_test(
  NAME gemm_allreduce_megacu_uses_runtime_arena
  COMMAND bash -lc
          "grep -R \"device_persistent\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/megacu && grep -R \"runtime::loop::block_tile\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/megacu")

add_test(
  NAME gemm_allreduce_megacu_has_no_manual_event_protocol
  COMMAND bash -lc
          "! grep -R \"task_event_tensor_i32\\|notify_task\\|wait_task\\|event_tensor.before\\|event_tensor.after(ctx, task\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/megacu")

add_test(
  NAME gemm_allreduce_megacu_dispatches_by_operator_slot
  COMMAND bash -lc
          "grep -R \"op_slot\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/megacu && ! grep -R \"task.value == 0\\|task.value == 2\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/megacu")

add_test(
  NAME gemm_allreduce_manual_megakernel_lives_only_in_baseline
  COMMAND bash -lc
          "grep -R \"manual_gemm_allreduce_megakernel\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/baseline && ! grep -R \"manual_gemm_allreduce_megakernel\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/megacu")
```

- [ ] **Step 3: Delete the legacy files**

Run:

```bash
rm examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_orchestrate_common.h
rm examples/cuda_nvshmem/gemm_allreduce/phased/megacu/CMakeLists.txt
rm examples/cuda_nvshmem/gemm_allreduce/phased/megacu/gemm_allreduce_phased_orchestrate.cc
rm examples/cuda_nvshmem/gemm_allreduce/phased/megacu/megacu_gemm_allreduce_phased.cu
```

- [ ] **Step 4: Run focused guard tests**

Run:

```bash
cmake --build build --target megacu_runtime_gemm_allreduce_recipe_contracts cuda_nvshmem_gemm_allreduce_megacu
ctest --test-dir build -R 'gemm_allreduce_megacu_|gemm_allreduce_manual_' --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A examples/cuda_nvshmem/gemm_allreduce
git commit -m "refactor: remove legacy gemm allreduce megacu proof path"
```

## Task 7: Update Example Documentation And Run Commands

**Files:**
- Create: `examples/cuda_nvshmem/gemm_allreduce/megacu/README.md`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/README.md`
- Modify: `docs/in_progress/general_runtime_linked_components_code_review.tmp.md`

- [ ] **Step 1: Write the Megacu variant README**

Create `examples/cuda_nvshmem/gemm_allreduce/megacu/README.md`:

```markdown
# GEMM-AllReduce Megacu Runtime Variant

This variant composes GEMM tile operators and AllReduce tile operators into one
Megacu mega-kernel through the common runtime path.

## Flow

```text
host calls cuda_nvshmem_gemm_allreduce_{host_orch,seeded_orch}(driver, raw args)
  -> shared recipe records EventTensor + GEMM task + sync task + AllReduce task
  -> host-orch seals records on host, or seeded-orch publishes records on device
  -> runtime::loop::block_tile asks explicit_asap for ready tasks
  -> tile_grid gives each block tile work for operator tasks
  -> attr_event_tensor_i32 lowers notify/wait attrs for sync readiness
  -> operator table dispatches by op_slot
```

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_gemm_allreduce_megacu
```

## Test

```sh
ctest --test-dir build -R 'runtime_gemm_allreduce|cuda_gemm_allreduce_correctness' --output-on-failure
```

The direct correctness test compares golden, manual baseline, Megacu
host-orch, and Megacu seeded-orch on one host and one CUDA device.

## Distributed Run Envelope

The distributed 1-host-2-device path uses the same host-callable functions and
driver ABI. MPI and Torch launch adapters construct the driver; the recipe and
runtime path are unchanged. Docker is the source-of-truth environment for those
runs in this PR.
```
```

- [ ] **Step 2: Update the family README**

Replace the stale active wording in `examples/cuda_nvshmem/gemm_allreduce/README.md` with:

```markdown
# CUDA+NVSHMEM GEMM-AllReduce

This example family validates the CUDA+NVSHMEM runtime path with a tiny
GEMM-AllReduce target. The Megacu variant is composed from operator tasks and
sync-only tasks. The handwritten mega-kernel is a baseline only.

## Layout

```text
common/             shared target ABI and Megacu recipe
golden/             Megacu-free golden entrypoints
phased/baseline/    handwritten CUDA/NVSHMEM manual mega-kernel baseline
megacu/             host-orch and seeded-orch Megacu runtime variant
```

## Execution

```text
orchestrate(driver, a, b, partial, out, events, problem)
  -> build_runtime_recipe(...)
  -> EventTensor ready object
  -> GEMM tile operator task
  -> sync-only wait task
  -> AllReduce tile operator task
  -> runtime-owned mega-kernel loop
```

## Usage

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_gemm_allreduce_megacu
ctest --test-dir build -R 'cuda_gemm_allreduce_correctness|runtime_gemm_allreduce' --output-on-failure
```
```

- [ ] **Step 3: Update the temporary code review dashboard**

In `docs/in_progress/general_runtime_linked_components_code_review.tmp.md`, update the GEMM-AllReduce row to say the tracer-bullet refit has host-orch/seeded-orch direct correctness evidence once Task 5 passes.

- [ ] **Step 4: Verify docs and guards**

Run:

```bash
rg "phased Megacu|active Megacu proof is phased|cuda_nvshmem_gemm_allreduce_phased_orchestrate|gemm_allreduce_orchestrate_common" examples/cuda_nvshmem/gemm_allreduce docs/in_progress/general_runtime_linked_components_code_review.tmp.md
```

Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add examples/cuda_nvshmem/gemm_allreduce/README.md \
  examples/cuda_nvshmem/gemm_allreduce/megacu/README.md \
  docs/in_progress/general_runtime_linked_components_code_review.tmp.md
git commit -m "docs: update gemm allreduce runtime example"
```

## Task 8: Final Focused Verification For The Refit

**Files:**
- No source files unless verification finds defects.

- [ ] **Step 1: Run focused build contracts**

Run:

```bash
cmake --build build --target \
  megacu_runtime_gemm_allreduce_recipe_contracts \
  megacu_runtime_gemm_allreduce_host_arena_contracts \
  cuda_nvshmem_gemm_allreduce_megacu
```

Expected: all targets build.

- [ ] **Step 2: Run focused tests**

Run:

```bash
ctest --test-dir build -R 'runtime_gemm_allreduce|cuda_gemm_allreduce_correctness|gemm_allreduce_megacu_|gemm_allreduce_manual_' --output-on-failure
```

Expected: PASS on a CUDA-capable machine. If the CUDA device is unavailable, record the exact CTest failure and run the compile-only targets plus non-CUDA guard tests.

- [ ] **Step 3: Run architecture scans**

Run:

```bash
! rg "task_event_tensor_i32|notify_task|wait_task|block_tile_runtime|gemm_allreduce_program|phased_megakernel|task.value == 0|task.value == 2" examples/cuda_nvshmem/gemm_allreduce/megacu
! rg "cuda_nvshmem_gemm_allreduce_phased_orchestrate|gemm_allreduce_orchestrate_common" examples/cuda_nvshmem/gemm_allreduce tests
```

Expected: both commands succeed.

- [ ] **Step 4: Run formatting and worktree checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: `git diff --check` has no output; status shows only intended committed history or a clean worktree.

- [ ] **Step 5: Commit verification notes if source changed during fixes**

If Step 1-4 required fixes, commit them:

```bash
git add <changed-files>
git commit -m "fix: complete gemm allreduce runtime refit"
```

If no files changed, do not create an empty commit.

## Scope Boundary For The Next Plan

This plan intentionally completes the GEMM-AllReduce direct tracer bullet first. The next implementation plan should reuse the same recipe/runtime shape for:

- GEMM-AllReduce 1-host-2-device through direct, MPI, and Torch adapters in Docker;
- GEMM-RS golden, baseline, host-orch, and seeded-orch;
- AG-GEMM golden, baseline, host-orch, and seeded-orch;
- tiny decode 1-host-1-device and 1-host-2-device host-orch and seeded-orch.

Do not start those examples before this GEMM-AllReduce refit passes its focused architecture guards.
