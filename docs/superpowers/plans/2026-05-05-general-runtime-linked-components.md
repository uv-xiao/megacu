# General Runtime-Linked Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the current PR architecture: runtime-owned execution models, runtime loops, EventTensor-owned sync behavior, required launch-adapter contracts, and three tile-operator examples.

**Architecture:** Runtime owns both the execution model and the device-side loop. Targets submit operator tasks and sync-only tasks with explicit attrs; scheduler, dispatcher, EventTensor, platform, and backend components are linked services called by the runtime loop. The Megacu example path must be composed from tile/range operators; handwritten large fused/persistent kernels are baseline-only.

**Tech Stack:** C++20 headers, CUDA `.cuh/.cu` contracts, CMake/CTest, CUDA+NVSHMEM examples, Docker-backed MPI and Torch launch validation.

---

## File Structure

- `include/megacu/runtime/task_arena.h`: compact task/event/dependency/region records shared by runtime execution models.
- `include/megacu/runtime/execution/host_orch.h`: host-built arena frame.
- `include/megacu/runtime/execution/seeded_orch.cuh`: device-built seeded arena execution model.
- `include/megacu/runtime/device_task_arena.cuh`: device-side publication helpers for seeded-orch.
- `include/megacu/runtime/device_persistent.cuh`: runtime-owned composition point.
- `include/megacu/runtime/loop/block_tile.cuh`: common block/tile runtime loop.
- `include/megacu/runtime/loop/grid_stride.cuh`: range loop for tiny decode and vector-style stages.
- `include/megacu/runtime/loop/single_work_item.cuh`: minimal loop for contract tests.
- `tests/build/runtime_execution_models_contracts.cc`: host-only arena and host-orch contracts.
- `tests/build/runtime_device_persistent_contract.cu`: CUDA compile/run contract for device_persistent.
- `tests/build/runtime_strategy_selection_contracts.cc`: host contract for selecting multiple dispatcher/scheduler/runtime candidates.
- `examples/cuda_nvshmem/gemm_reduce_scatter/`: GEMM-RS golden/baseline/Megacu example.
- `examples/cuda_nvshmem/allgather_gemm/`: AG-GEMM golden/baseline/Megacu example.
- `examples/cuda_nvshmem/tiny_decode_pipeline/`: tiny decode golden/baseline/Megacu example.
- `docker/cuda_nvshmem/`: required CUDA+NVSHMEM+MPI+PyTorch execution image.
- `tools/cuda_nvshmem/`: direct, MPI, and Torch run helpers.

## Task 1: Runtime Composition

**Files:**
- Create: `include/megacu/runtime/device_persistent.cuh`
- Create: `include/megacu/runtime/loop/block_tile.cuh`
- Modify: `include/megacu/runtime/block_tile_runtime.cuh`
- Test: `tests/build/runtime_device_persistent_contract.cu`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`

- [x] **Step 1: Write the failing CUDA contract**

Add `tests/build/runtime_device_persistent_contract.cu` with this shape:

```cpp
#include <cassert>
#include <cuda_runtime.h>
#include <megacu/runtime/device_persistent.cuh>
#include <megacu/runtime/loop/block_tile.cuh>
#include <megacu/runtime/task_arena.h>

struct fake_context {};
struct fake_execution {
  megacu::runtime::task_arena_view arena;
  __device__ megacu::runtime::task_arena_view bind(fake_context) const {
    return arena;
  }
  __device__ void construct(fake_context, megacu::runtime::task_arena_view) const {}
};
struct fake_scheduler {};
struct fake_dispatcher {};
struct fake_event_tensor {};
struct fake_operators {};
struct fake_loop {
  int *marker = nullptr;
  template <class Context, class Arena, class Scheduler, class Dispatcher,
            class EventTensor, class Operators>
  __device__ void run(Context, Arena, Scheduler, Dispatcher, EventTensor,
                      Operators) const {
    if (threadIdx.x == 0) *marker = 7;
  }
};
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cmake -S . -B build
cmake --build build --target megacu_runtime_device_persistent_contract
```

Expected: FAIL because `megacu/runtime/device_persistent.cuh` or `megacu/runtime/loop/block_tile.cuh` is missing.

- [x] **Step 3: Add minimal runtime composition**

Implement `include/megacu/runtime/device_persistent.cuh`:

```cpp
#pragma once
namespace megacu::runtime {
template <class ExecutionModel, class Loop, class Scheduler, class Dispatcher,
          class EventTensor, class Operators>
struct device_persistent {
  ExecutionModel execution;
  Loop loop;
  Scheduler scheduler;
  Dispatcher dispatcher;
  EventTensor event_tensor;
  Operators operators;
  template <class Context> __device__ void operator()(Context ctx) const {
    auto arena = execution.bind(ctx);
    execution.construct(ctx, arena);
    loop.run(ctx, arena, scheduler, dispatcher, event_tensor, operators);
  }
};
}
```

Implement `include/megacu/runtime/loop/block_tile.cuh` by moving the loop body from `runtime::device::block_tile_runtime` into `runtime::loop::block_tile::run(...)`.

- [x] **Step 4: Keep compatibility path**

Modify `include/megacu/runtime/block_tile_runtime.cuh` so `runtime::device::block_tile_runtime::operator()` delegates to `runtime::loop::block_tile{}.run(...)`. Existing examples must still compile.

- [x] **Step 5: Run test to verify it passes**

Run:

```bash
cmake --build build --target megacu_runtime_device_persistent_contract
ctest --test-dir build -R runtime_device_persistent_contract --output-on-failure
```

Expected: build succeeds and CTest passes if a CUDA device is available. If no CUDA device is available, record the runtime skip risk and keep the build target as the evidence.

## Task 2: Seeded-Orch Publication

**Files:**
- Create: `include/megacu/runtime/device_task_arena.cuh`
- Create: `include/megacu/runtime/execution/seeded_orch.cuh`
- Modify: `tests/build/runtime_device_persistent_contract.cu`
- Modify: `tests/build/runtime_execution_models_contracts.cc`

- [x] **Step 1: Write failing seeded-orch contract**

Extend the CUDA contract with a recipe that publishes one event, one producer task, one sync-only task, and one consumer task into a device-visible arena:

```cpp
struct seeded_recipe {
  template <class Orch>
  __device__ megacu::status operator()(Orch &orch) const {
    auto ready = orch.event_tensor(megacu::runtime::attrs(
        megacu::runtime::event_tensor::shape(1, 1),
        megacu::runtime::event_tensor::wait_count(1)));
    auto producer = orch.submit(megacu::runtime::operator_slot{3},
        megacu::runtime::attrs(megacu::runtime::event_tensor::notify(ready)));
    auto sync = orch.sync(megacu::runtime::attrs(
        megacu::runtime::scheduler::depends_on(producer),
        megacu::runtime::event_tensor::wait(ready)));
    orch.submit(megacu::runtime::operator_slot{4},
        megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(sync)));
    return orch.seal();
  }
};
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target megacu_runtime_device_persistent_contract
```

Expected: FAIL because `runtime::execution::seeded_orch` and device publication helpers do not exist.

- [x] **Step 3: Implement device task arena helpers**

Add `device_task_arena.cuh` with a device facade that writes `device_task_record`, `device_event_tensor_record`, dependencies, and one sealed `arena_region`. It must return `status_code::invalid_argument` on capacity overflow and must not infer dependencies.

- [x] **Step 4: Implement seeded-orch execution model**

Add `execution::seeded_orch::model<Recipe>`:

```cpp
template <class Recipe>
struct model {
  task_arena_view arena;
  Recipe recipe;
  status *device_status = nullptr;

  template <class Context>
  __device__ task_arena_view bind(Context) const { return arena; }

  template <class Context>
  __device__ void construct(Context ctx, task_arena_view view) const {
    if (ctx.block_id() != 0 || ctx.thread_id() != 0) return;
    device_orch orch{view};
    auto result = recipe(orch);
    if (device_status != nullptr) *device_status = result;
  }
};
```

- [x] **Step 5: Run test to verify it passes**

Run:

```bash
cmake --build build --target megacu_runtime_device_persistent_contract megacu_runtime_execution_models_contracts
ctest --test-dir build -R 'runtime_device_persistent_contract|runtime_execution_models_contracts' --output-on-failure
```

Expected: seeded-orch publishes region 0, epoch 0, task count 3, event count 1, and matching dependency refs.

## Task 3: Strategy Selection Contracts

**Files:**
- Create: `include/megacu/runtime/loop/grid_stride.cuh`
- Create: `include/megacu/runtime/loop/single_work_item.cuh`
- Create: `tests/build/runtime_strategy_selection_contracts.cc`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`

- [x] **Step 1: Write failing strategy-selection test**

Create a C++ build test that includes the new loop headers and asserts the types exist:

```cpp
#include <type_traits>
#include <megacu/runtime/loop/block_tile.cuh>
#include <megacu/runtime/loop/grid_stride.cuh>
#include <megacu/runtime/loop/single_work_item.cuh>
int main() {
  static_assert(std::is_empty_v<megacu::runtime::loop::block_tile>);
  static_assert(std::is_empty_v<megacu::runtime::loop::grid_stride>);
  static_assert(std::is_empty_v<megacu::runtime::loop::single_work_item>);
  return 0;
}
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target megacu_runtime_strategy_selection_contracts
```

Expected: FAIL because the new loop headers do not exist.

- [x] **Step 3: Implement minimal loop candidates**

Add empty-but-callable `grid_stride` and `single_work_item` loop structs with the same `run(ctx, arena, scheduler, dispatcher, event_tensor, operators)` shape as `block_tile`. Keep behavior minimal and generic.

- [x] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target megacu_runtime_strategy_selection_contracts
ctest --test-dir build -R runtime_strategy_selection_contracts --output-on-failure
```

Expected: test passes and no example code calls `scheduler.run`.

## Task 4: Tile-Operator Example Skeletons

**Files:**
- Create: `examples/cuda_nvshmem/gemm_reduce_scatter/README.md`
- Create: `examples/cuda_nvshmem/gemm_reduce_scatter/CMakeLists.txt`
- Create: `examples/cuda_nvshmem/gemm_reduce_scatter/common/.gitkeep`
- Create: `examples/cuda_nvshmem/gemm_reduce_scatter/golden/.gitkeep`
- Create: `examples/cuda_nvshmem/gemm_reduce_scatter/baseline/.gitkeep`
- Create: `examples/cuda_nvshmem/gemm_reduce_scatter/megacu/gemm_reduce_scatter_megacu.cu`
- Create: `examples/cuda_nvshmem/allgather_gemm/README.md`
- Create: `examples/cuda_nvshmem/allgather_gemm/CMakeLists.txt`
- Create: `examples/cuda_nvshmem/allgather_gemm/common/.gitkeep`
- Create: `examples/cuda_nvshmem/allgather_gemm/golden/.gitkeep`
- Create: `examples/cuda_nvshmem/allgather_gemm/baseline/.gitkeep`
- Create: `examples/cuda_nvshmem/allgather_gemm/megacu/allgather_gemm_megacu.cu`
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/README.md`
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/CMakeLists.txt`
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/common/.gitkeep`
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/golden/.gitkeep`
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/baseline/.gitkeep`
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_megacu.cu`
- Modify: `examples/cuda_nvshmem/CMakeLists.txt`
- Modify: `examples/cuda_nvshmem/README.md`

- [x] **Step 1: Write failing CMake path checks**

Add these tests to `examples/cuda_nvshmem/CMakeLists.txt` before adding the
directories:

```cmake
function(megacu_require_example_layout test_name example_dir megacu_source)
  add_test(
    NAME ${test_name}
    COMMAND ${CMAKE_COMMAND} -E env
            bash -lc
            "test -f ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/${example_dir}/README.md && test -f ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/${example_dir}/${megacu_source}")
endfunction()

megacu_require_example_layout(
  gemm_rs_example_layout
  gemm_reduce_scatter
  megacu/gemm_reduce_scatter_megacu.cu)
megacu_require_example_layout(
  ag_gemm_example_layout
  allgather_gemm
  megacu/allgather_gemm_megacu.cu)
megacu_require_example_layout(
  tiny_decode_example_layout
  tiny_decode_pipeline
  megacu/tiny_decode_megacu.cu)
```

- [x] **Step 2: Run checks to verify they fail**

Run:

```bash
cmake -S . -B build
ctest --test-dir build -R 'gemm_rs_example_layout|ag_gemm_example_layout|tiny_decode_example_layout' --output-on-failure
```

Expected: FAIL because directories and files are missing.

- [x] **Step 3: Add example skeletons**

Create each example with `README.md`, `CMakeLists.txt`, `common/`, `golden/`,
`baseline/`, and `megacu/`. Each `CMakeLists.txt` should be minimal:

```cmake
add_library(cuda_nvshmem_<example>_megacu OBJECT
  megacu/<source>.cu)
target_link_libraries(cuda_nvshmem_<example>_megacu PRIVATE
  megacu_headers)
```

Each Megacu source should be a placeholder source that compiles:

```cpp
#include <megacu/runtime.h>

extern "C" int megacu_<example>_skeleton() {
  return 1;
}
```

Each README must include these points:

- what the example demonstrates;
- layout: `common/`, `golden/`, `baseline/`, `megacu/`;
- Megacu path submits tile/range operators as tasks;
- scheduler, dispatcher, runtime, and EventTensor synchronize and issue tasks;
- baseline is the only place for handwritten large fused/persistent kernels;
- correctness implementation is in later tasks.

Update `examples/cuda_nvshmem/CMakeLists.txt` to add the three subdirectories
after the layout tests. Update `examples/cuda_nvshmem/README.md` to list the
three examples.

- [x] **Step 4: Run checks to verify they pass**

Run:

```bash
cmake -S . -B build
ctest --test-dir build -R 'gemm_rs_example_layout|ag_gemm_example_layout|tiny_decode_example_layout' --output-on-failure
```

Expected: layout checks pass.

## Task 5: Tiny Decode Correctness Example

**Files:**
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/common/tiny_decode.h`
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/golden/tiny_decode_golden.cc`
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/baseline/tiny_decode_baseline.cu`
- Create: `examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_megacu.cu`
- Modify: `examples/cuda_nvshmem/tiny_decode_pipeline/CMakeLists.txt`
- Create: `tests/runtime/tiny_decode_correctness.cc`

Define this public C ABI in `common/tiny_decode.h`:

```cpp
#pragma once

#include <cstddef>

namespace megacu::examples::tiny_decode {

inline constexpr std::size_t hidden_size = 4;
inline constexpr std::size_t mlp_size = 4;
inline constexpr std::size_t vocab_size = 3;

struct buffers {
  float hidden[hidden_size];
  float norm[hidden_size];
  float projection[hidden_size];
  float residual[hidden_size];
  float mlp[mlp_size];
  float logits[vocab_size];
};

void seed_inputs(buffers &state);
bool nearly_equal(buffers const &lhs, buffers const &rhs);

} // namespace megacu::examples::tiny_decode

extern "C" int megacu_tiny_decode_golden(
    megacu::examples::tiny_decode::buffers *state);
extern "C" int megacu_tiny_decode_baseline(
    megacu::examples::tiny_decode::buffers *state);
extern "C" int megacu_tiny_decode_megacu(
    megacu::examples::tiny_decode::buffers *state);
```

Use the same deterministic math in all three variants:

```text
norm[i] = hidden[i] * 0.5
projection[i] = norm[i] + (i + 1)
residual[i] = projection[i] + hidden[i]
mlp[i] = residual[i] * residual[i] * 0.25
logits[j] = sum_i mlp[i] * ((i + 1) * (j + 1) * 0.125)
```

The Megacu variant must not implement the whole pipeline as one operator.
Submit one operator task per stage through `runtime::phase`: norm, projection,
residual, mlp, logits. Use explicit `scheduler::depends_on(...)` attrs between
stages and run the phase. It is acceptable for this first tiny decode
correctness test to use the host `phase` path; later tasks can move it to a
CUDA mega-kernel once the example’s numeric contract exists.

- [x] **Step 1: Write failing golden/baseline/Megacu correctness test**

Create `tests/runtime/tiny_decode_correctness.cc`:

```cpp
#include <cassert>

#include "examples/cuda_nvshmem/tiny_decode_pipeline/common/tiny_decode.h"

int main() {
  megacu::examples::tiny_decode::buffers golden{};
  megacu::examples::tiny_decode::buffers baseline{};
  megacu::examples::tiny_decode::buffers megacu{};

  megacu::examples::tiny_decode::seed_inputs(golden);
  baseline = golden;
  megacu = golden;

  assert(megacu_tiny_decode_golden(&golden) == 0);
  assert(megacu_tiny_decode_baseline(&baseline) == 0);
  assert(megacu_tiny_decode_megacu(&megacu) == 0);

  assert(megacu::examples::tiny_decode::nearly_equal(golden, baseline));
  assert(megacu::examples::tiny_decode::nearly_equal(golden, megacu));
  return 0;
}
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target cuda_nvshmem_tiny_decode_correctness
ctest --test-dir build -R tiny_decode_correctness --output-on-failure
```

Expected: FAIL because the correctness target or implementation functions do
not exist.

- [x] **Step 3: Implement golden and baseline**

Implement `seed_inputs`, `nearly_equal`, and `megacu_tiny_decode_golden` in
`golden/tiny_decode_golden.cc`. Implement `megacu_tiny_decode_baseline` in
`baseline/tiny_decode_baseline.cu` using the same stage math.

- [x] **Step 4: Implement Megacu path**

Implement `megacu_tiny_decode_megacu` in `megacu/tiny_decode_megacu.cu`.
Define one small native stage function per stage and submit those functions as
separate Megacu tasks with explicit dependency attrs:

```cpp
auto norm = phase.submit(runtime::op("tiny_decode_norm", &tiny_decode_norm), state);
auto projection = phase.submit(runtime::op("tiny_decode_projection", &tiny_decode_projection), state,
    runtime::attrs(runtime::scheduler::depends_on(norm)));
auto residual = phase.submit(runtime::op("tiny_decode_residual", &tiny_decode_residual), state,
    runtime::attrs(runtime::scheduler::depends_on(projection)));
auto mlp = phase.submit(runtime::op("tiny_decode_mlp", &tiny_decode_mlp), state,
    runtime::attrs(runtime::scheduler::depends_on(residual)));
phase.submit(runtime::op("tiny_decode_logits", &tiny_decode_logits), state,
    runtime::attrs(runtime::scheduler::depends_on(mlp)));
```

The function should return `0` only when `phase.run()` returns `status_code::ok`.

- [x] **Step 5: Run test to verify it passes**

Run:

```bash
cmake --build build --target cuda_nvshmem_tiny_decode_correctness
ctest --test-dir build -R tiny_decode_correctness --output-on-failure
```

Expected: golden, baseline, and Megacu outputs match.

## Task 6: MPI/Torch Docker Adapter Contracts

**Files:**
- Modify: `docker/cuda_nvshmem/Dockerfile`
- Modify: `docker/cuda_nvshmem/README.md`
- Create: `tools/cuda_nvshmem/run_direct.sh`
- Create: `tools/cuda_nvshmem/run_mpi.sh`
- Create: `tools/cuda_nvshmem/run_torch.py`
- Create: `include/megacu/adapters/mpi_cuda_nvshmem.h`
- Create: `include/megacu/adapters/torch_cuda_nvshmem.h`
- Create: `tests/build/adapter_contracts.cc`
- Modify: `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt`
- Modify: `tools/cuda_nvshmem/README.md`

Adapter API:

```cpp
namespace megacu::adapters::mpi_cuda_nvshmem {

struct launch_facts {
  std::int32_t rank = 0;
  std::int32_t world_size = 1;
  std::int32_t local_rank = 0;
};

megacu::cuda_nvshmem::driver_view make_driver(
    launch_facts facts, void *stream = nullptr);

} // namespace megacu::adapters::mpi_cuda_nvshmem

namespace megacu::adapters::torch_cuda_nvshmem {

struct launch_facts {
  std::int32_t rank = 0;
  std::int32_t world_size = 1;
  std::int32_t local_rank = 0;
};

megacu::cuda_nvshmem::driver_view make_driver(
    launch_facts facts, void *stream = nullptr);

} // namespace megacu::adapters::torch_cuda_nvshmem
```

Both adapters should only normalize launch facts into `driver_view`:

```text
driver.launch.stream = stream
driver.launch.device_ordinal = facts.local_rank
driver.team.team_my_pe = facts.rank
driver.team.team_n_pes = facts.world_size
driver.team.world_my_pe = facts.rank
driver.team.world_n_pes = facts.world_size
driver.team.cuda_device_ordinal = facts.local_rank
```

No adapter may select scheduler order, dispatcher placement, runtime loop, or
operator tasks.

- [x] **Step 1: Write failing adapter smoke checks**

Create `tests/build/adapter_contracts.cc`:

```cpp
#include <cassert>

#include <megacu/adapters/mpi_cuda_nvshmem.h>
#include <megacu/adapters/torch_cuda_nvshmem.h>

int main() {
  auto mpi = megacu::adapters::mpi_cuda_nvshmem::make_driver(
      {.rank = 2, .world_size = 4, .local_rank = 1},
      reinterpret_cast<void *>(0x1));
  assert(mpi.launch.stream == reinterpret_cast<void *>(0x1));
  assert(mpi.launch.device_ordinal == 1);
  assert(mpi.team.team_my_pe == 2);
  assert(mpi.team.team_n_pes == 4);
  assert(mpi.team.world_my_pe == 2);
  assert(mpi.team.world_n_pes == 4);
  assert(mpi.team.cuda_device_ordinal == 1);

  auto torch = megacu::adapters::torch_cuda_nvshmem::make_driver(
      {.rank = 3, .world_size = 8, .local_rank = 0});
  assert(torch.launch.stream == nullptr);
  assert(torch.launch.device_ordinal == 0);
  assert(torch.team.team_my_pe == 3);
  assert(torch.team.team_n_pes == 8);
  assert(torch.team.world_my_pe == 3);
  assert(torch.team.world_n_pes == 8);
  assert(torch.team.cuda_device_ordinal == 0);
  return 0;
}
```

Add CMake targets/tests. The final implementation keeps a local shape contract
and separate launch-environment contracts:

```cmake
add_executable(megacu_adapter_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/adapter_contracts.cc)
target_link_libraries(megacu_adapter_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(NAME adapter_local_shape_contract COMMAND megacu_adapter_contracts)
add_test(NAME mpi_adapter_env_contract COMMAND ...)
add_test(NAME torch_adapter_env_contract COMMAND ...)
```

Add CTest script checks requiring these files:

```text
tools/cuda_nvshmem/run_direct.sh
tools/cuda_nvshmem/run_mpi.sh
tools/cuda_nvshmem/run_torch.py
```

- [x] **Step 2: Run checks to verify they fail**

Run:

```bash
cmake -S . -B build
ctest --test-dir build -R 'adapter_local_shape_contract|mpi_adapter_env_contract|torch_adapter_env_contract' --output-on-failure
```

Expected: FAIL because adapters are missing.

- [x] **Step 3: Implement thin adapters**

Implement the two headers as header-only helpers using the API above.

Update `docker/cuda_nvshmem/Dockerfile` so PyTorch is required in the Docker
image:

```dockerfile
RUN python3 -m pip install --no-cache-dir \
    nvidia-nvshmem-cu12==3.6.5 \
    torch
```

Create helper scripts:

```bash
# tools/cuda_nvshmem/run_direct.sh
#!/usr/bin/env bash
set -euo pipefail
build_dir="${MEGACU_BUILD_DIR:-build}"
cmake --build "${build_dir}" --target cuda_nvshmem_tiny_decode_correctness
ctest --test-dir "${build_dir}" -R 'tiny_decode_correctness' --output-on-failure
```

```bash
# tools/cuda_nvshmem/run_mpi.sh
#!/usr/bin/env bash
set -euo pipefail
build_dir="${MEGACU_BUILD_DIR:-build}"
mpi_np="${MEGACU_MPI_NP:-2}"
cmake --build "${build_dir}" --target megacu_adapter_contracts
mpirun -np "${mpi_np}" "$<do not use generator expressions here>"
```

For the actual script, avoid CMake generator expressions. Use the built binary
path `"${build_dir}/examples/cuda_nvshmem/gemm_allreduce/megacu_adapter_contracts"`.

```python
# tools/cuda_nvshmem/run_torch.py
#!/usr/bin/env python3
import os
import subprocess
import torch
import torch.distributed as dist

# The actual script requires torchrun-provided RANK, WORLD_SIZE, LOCAL_RANK,
# initializes torch.distributed, and runs the adapter contract in torch mode.
```

Update `tools/cuda_nvshmem/README.md` and `docker/cuda_nvshmem/README.md` to
state MPI and Torch are required PR dependencies in Docker.

- [x] **Step 4: Run checks to verify they pass**

Run:

```bash
cmake -S . -B build
ctest --test-dir build -R 'adapter_local_shape_contract|mpi_adapter_env_contract|mpi_adapter_requires_local_rank|torch_adapter_env_contract|cuda_nvshmem_script_layout' --output-on-failure
```

Expected: adapter contract checks pass locally. Full MPI/Torch execution evidence is Docker-only.

## Final Verification

- [x] Run `git diff --check`.
- [x] Run the repo documentation policy grep for forbidden stale terms from a
  shell-local pattern, not by storing the pattern text in committed docs.
  Expected: no matches.

- [x] Run local targeted tests:

```bash
cmake -S . -B build
cmake --build build --target megacu_runtime_execution_models_contracts megacu_runtime_components_contracts megacu_runtime_linked_surface_contract
ctest --test-dir build -R 'runtime_execution_models_contracts|runtime_components_contracts|runtime_linked_surface_contract|megacu_event_tensor|megakernel_running_logic_is_common' --output-on-failure
```

- [ ] Run Docker verification once the image and examples exist:

```bash
docker build -f docker/cuda_nvshmem/Dockerfile -t megacu-cuda-nvshmem .
tools/cuda_nvshmem/run_direct.sh
tools/cuda_nvshmem/run_mpi.sh
torchrun --standalone --nproc_per_node=2 tools/cuda_nvshmem/run_torch.py
```

Expected: direct, MPI, and Torch paths report correctness for the required examples.
