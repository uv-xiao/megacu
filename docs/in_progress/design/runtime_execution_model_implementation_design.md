# Runtime Execution Model Implementation Design

Status: active implementation design for the general runtime-linked components
PR. This document turns `execution_model_study.md` into concrete code changes.

## Goal

Implement two runtime execution models in this PR:

- `host-orch`: host orch builds and seals one arena region before launching the
  mega-kernel.
- `seeded-orch`: host seeds launch facts, then a device orch path builds and
  seals at least one arena region inside the mega-kernel.

Both models must use the same runtime-owned composition shape:

```text
runtime::device_persistent<
  runtime::execution::<model>,
  runtime::loop::block_tile>
```

The execution model is not a top-level `ConfigureTarget` component. It belongs
inside runtime, beside the runtime loop.

## Current Implementation Gap

The branch already has useful pieces:

- host-side `runtime::phase` records tasks and EventTensor attrs;
- device-side `runtime::device::block_tile_runtime` owns a common loop;
- `scheduler::device::explicit_asap` iterates task ids;
- `dispatcher::device::tile_grid` maps blocks to tile work;
- `backend::nvshmem::cuda::task_event_tensor_i32` lowers notify/wait around
  tasks;
- the CUDA+NVSHMEM GEMM+AllReduce proof launches one mega-kernel.

The mismatch is that task ids, EventTensor behavior, task count, and operator
dispatch are still assembled manually in the example launcher. That makes the
example look like a handwritten mega-kernel wrapper instead of Megacu composing
operators into a runtime-owned execution model. The next implementation must
remove that pattern from Megacu examples: examples should submit tile/range
operator tasks, and the linked runtime, scheduler, dispatcher, and EventTensor
components should form the mega-kernel behavior.

## Target Code Shape

### Runtime Namespaces

Add explicit runtime namespaces:

```text
include/megacu/runtime/execution/host_orch.h
include/megacu/runtime/execution/seeded_orch.cuh
include/megacu/runtime/loop/block_tile.cuh
include/megacu/runtime/device_persistent.cuh
include/megacu/runtime/task_arena.h
include/megacu/runtime/device_task_arena.cuh
```

`block_tile_runtime.cuh` can either move to `runtime/loop/block_tile.cuh` or
remain as a compatibility wrapper that aliases the new name.

### Runtime Composition

The device runtime is the owner of both execution model and loop:

```cpp
template <class ExecutionModel, class Loop, class Scheduler, class Dispatcher,
          class EventTensor, class Operators>
struct device_persistent {
  ExecutionModel execution;
  Loop loop;
  Scheduler scheduler;
  Dispatcher dispatcher;
  EventTensor event_tensor;
  Operators operators;

  template <class Context>
  __device__ void operator()(Context ctx) const {
    auto arena = execution.bind(ctx);
    execution.construct(ctx, arena);
    loop.run(ctx, arena, scheduler, dispatcher, event_tensor, operators);
  }
};
```

For `host-orch`, `construct` is a no-op because the arena is already sealed.

For the first `seeded-orch`, `construct` uses one control writer to publish one
bounded region, seals the region, then the loop consumes it. Later runtimes can
allow construction and execution to proceed concurrently, but this PR should
first prove device-side construction, publication, sealing, and distributed rank
agreement.

## Arena Records

Host and device execution models need a compact common record shape. The device
arena must not contain `std::function`, host-only pointers, or generic argument
blobs.

```cpp
struct device_task_record {
  task_kind kind;
  std::uint16_t op_slot;
  std::uint16_t dep_count;
  std::uint16_t first_dep;
  attr_set attrs;
};

struct device_event_tensor_record {
  event_tensor_ref ref;
  attr_set attrs;
};

struct arena_region {
  std::uint32_t epoch;
  std::uint32_t first_task;
  std::uint32_t task_count;
  std::uint32_t first_event;
  std::uint32_t event_count;
  std::uint32_t sealed;
};

struct task_arena_view {
  device_task_record *tasks;
  device_event_tensor_record *events;
  task_ref *deps;
  arena_region *regions;
  std::uint32_t task_capacity;
  std::uint32_t event_capacity;
  std::uint32_t dep_capacity;
  std::uint32_t region_capacity;
};
```

The first implementation should use one region and one epoch:

```text
region 0
  epoch = 0
  tasks = generated task records
  events = generated EventTensor records
  sealed = true
```

Capacity overflow is an explicit error. Megacu does not fallback to host
dispatch or infer missing records.

## Shared Recipe Pattern

The same target-level task construction logic should drive both execution
models. Do not write one algorithm for `host-orch` and another for
`seeded-orch`.

Use an orch-concept recipe. The exact operator slots differ by target, but the
pattern is shared:

```cpp
template <class Orch>
MEGACU_HOST_DEVICE megacu::status gemm_rs_recipe(
    Orch &orch, gemm_rs_runtime_args args) {
  auto ready = orch.event_tensor(attrs(
      event_tensor::shape(args.m_tiles, args.n_tiles),
      event_tensor::wait_count(args.team_size),
      cuda_nvshmem::event_tensor::symmetric_storage(args.events),
      cuda_nvshmem::event_tensor::scope::team{}));

  auto gemm = orch.submit(op_slot::gemm_rs_gemm_tile, args.produce_args,
                          attrs(dispatcher::tile_grid(args.m_tiles,
                                                      args.n_tiles),
                                event_tensor::notify(ready)));

  auto ready_task = orch.sync(attrs(scheduler::depends_on(gemm),
                                    event_tensor::wait(ready)));

  orch.submit(op_slot::reduce_scatter_tile, args.consume_args,
              attrs(scheduler::depends_on(ready_task)));

  return {};
}
```

`host-orch` calls this recipe on the host and writes records into a host arena
that is copied or lowered to device-visible records.

`seeded-orch` stores target runtime args as seed data, then calls the same
recipe from the device control path using a device orch object that publishes
into the device arena.

The same structure should be used for AG-GEMM and tiny decode. AG-GEMM swaps
the producer/consumer roles to communication-producer tasks followed by GEMM
tile consumer tasks. Tiny decode uses local range/tile operator stages, with
sync-only tasks where stage readiness needs to be represented explicitly.

## Operator Dispatch

Operator kernels should use a common thin API. They should not call scheduler,
dispatcher, EventTensor, CUDA stream, or NVSHMEM sync APIs manually.

The runtime loop invokes an operator table by slot:

```cpp
struct gemm_rs_operator_table {
  gemm_rs_gemm_tile_task gemm;
  reduce_scatter_tile_task reduce_scatter;

  template <class Context, class Work>
  __device__ void invoke(Context ctx, op_slot slot, Work work) const {
    switch (slot.value) {
    case op_slot::gemm_rs_gemm_tile:
      gemm(ctx, work.tile_id);
      break;
    case op_slot::reduce_scatter_tile:
      reduce_scatter(ctx, work.tile_id);
      break;
    }
  }
};
```

The task record stores `op_slot`, not an example-specific branch on hardcoded
task ids.

## Host-Orch Implementation

Files:

- `include/megacu/runtime/execution/host_orch.h`
- `src/runtime/host_orch.cc`
- `tests/build/runtime_execution_models_contracts.cc`

Behavior:

1. Host creates a `host_orch::frame`.
2. Target recipe calls `event_tensor`, `submit`, and `sync`.
3. The frame assigns deterministic task ids and EventTensor ids.
4. The frame creates one region with epoch `0`.
5. `seal()` freezes the arena and rejects further submissions.
6. The launch path creates `device_persistent<host_orch, block_tile>`.
7. The runtime loop consumes the sealed arena.

Required tests:

- submitting after seal returns an explicit error;
- scheduler sees only published records;
- event wait remains on the sync task, not the consumer task;
- operator task records carry op slots, not hardcoded task-id behavior;
- region count is one and epoch is zero.

## Seeded-Orch Implementation

Files:

- `include/megacu/runtime/execution/seeded_orch.cuh`
- `include/megacu/runtime/device_task_arena.cuh`
- `tests/build/runtime_execution_models_contracts.cc`
- CUDA runtime smoke/example tests under
  `examples/cuda_nvshmem/gemm_reduce_scatter/megacu/` and
  `examples/cuda_nvshmem/allgather_gemm/megacu/`.

Behavior:

1. Host prepares seed args, operator table, backend/team facts, EventTensor
   storage, and arena capacity.
2. Host launches one mega-kernel with
   `device_persistent<seeded_orch, block_tile>`.
3. One device control writer initializes region `0`, epoch `0`.
4. Device orch calls the same target recipe with a device orch facade.
5. Device orch publishes task and EventTensor records into the arena.
6. Device orch seals the region.
7. Worker path waits for the region to become sealed, then the runtime loop
   executes published records.

The first PR implementation can serialize construction before execution inside
the mega-kernel. It still proves the key ownership point: task records are
created inside the runtime execution model, not manually assembled in the
example launcher. A later loop can consume published tasks before the region is
fully sealed if we choose a more dynamic policy.

Required tests:

- seeded orch publishes at least one region on device;
- region `0` has epoch `0` and sealed state;
- capacity overflow reports an explicit error in device-visible status;
- normal operator tasks do not publish additional tasks;
- `host-orch` and `seeded-orch` produce equivalent task/event structure for the
  same recipe.

## CUDA+NVSHMEM Example Shape

Each distributed example should expose two Megacu variants under the same
example family:

```text
examples/cuda_nvshmem/gemm_reduce_scatter/megacu/
  host_orch/
  seeded_orch/
  common/

examples/cuda_nvshmem/allgather_gemm/megacu/
  host_orch/
  seeded_orch/
  common/

examples/cuda_nvshmem/tiny_decode_pipeline/megacu/
  host_orch/
  seeded_orch/
  common/
```

If the repository keeps a flatter layout for now, target names must still make
the distinction explicit:

```text
cuda_nvshmem_gemm_rs_host_orch
cuda_nvshmem_gemm_rs_seeded_orch
cuda_nvshmem_ag_gemm_host_orch
cuda_nvshmem_ag_gemm_seeded_orch
cuda_nvshmem_tiny_decode_host_orch
cuda_nvshmem_tiny_decode_seeded_orch
```

Both variants should use the same recipe and operator table. They may differ in
how the arena is built:

- `host-orch`: host frame builds and seals records before launch.
- `seeded-orch`: device orch builds and seals records inside the mega-kernel.

The distributed examples must run:

- one host / one GPU;
- one host / two GPUs through CUDA+NVSHMEM launch infrastructure.

The manual mega-kernel, when useful, remains baseline code under `baseline/`.
The Megacu variant must be composed from submitted tile/range operators.

## Implementation Slices

### Slice 3: Arena Records And Host-Orch

Inputs:

- existing `runtime::phase`;
- existing `linked_task` and `linked_event_tensor`;
- current host build tests.

Outputs:

- compact device record structs;
- `host_orch` frame with one region/epoch;
- target validation that `host-orch` is runtime-owned;
- host contract tests.

Verification:

- `cmake --build build --target megacu_runtime_components_contracts`
- new `runtime_execution_models_contracts` build test;
- grep guard rejecting `build_run::` in target configuration docs/examples.

### Slice 4: Runtime Composition And Loop Rename

Inputs:

- `include/megacu/runtime/block_tile_runtime.cuh`;
- current CUDA+NVSHMEM Megacu launcher.

Outputs:

- `runtime::device_persistent<ExecutionModel, Loop, ...>`;
- `runtime::loop::block_tile`;
- compatibility path for existing `block_tile_runtime` only if needed during
  migration;
- example instantiates runtime-owned execution model plus loop.

Verification:

- build test proving `runtime::execution::host_orch` and
  `runtime::execution::seeded_orch` are nested under runtime;
- grep guard rejecting top-level `build_run::` in code.

### Slice 5: Seeded-Orch Device Publication

Inputs:

- common recipe;
- device task arena;
- operator table.

Outputs:

- `seeded_orch` device execution model;
- one-region/one-epoch device publication;
- device-visible status for capacity and unsupported attrs;
- example target using seeded device construction.

Verification:

- CUDA build for seeded-orch target;
- local single-GPU seeded-orch run;
- distributed one-host/two-GPU seeded-orch run;
- task/event structure equivalence check against host-orch for the same recipe.

### Slice 6: Example And Adapter Validation

Inputs:

- GEMM-RS, AG-GEMM, and tiny decode golden/manual baselines;
- required MPI/Torch Docker environment from this PR's broader scope.

Outputs:

- host-orch and seeded-orch Megacu example binaries;
- direct/local run commands documented;
- MPI and Torch launch adapter run commands documented;
- README explains environment setup and binary execution.

Verification:

- Docker build with CUDA, NVSHMEM, MPI, and PyTorch;
- golden, baseline, host-orch, and seeded-orch correctness comparison for
  GEMM-RS, AG-GEMM, and tiny decode;
- one host / one GPU and one host / two GPU execution evidence.

## Rejected Paths

- Do not add `build_run` as a top-level `ConfigureTarget` component.
- Do not let examples manually assemble task ids for EventTensor notify/wait.
- Do not let normal operator tasks publish new tasks.
- Do not add TensorMap-style dependency inference.
- Do not use CUDA streams for internal task synchronization inside the
  mega-kernel.
- Do not add fallback checking or repair for missing dependencies.

## Acceptance Criteria

This implementation design is satisfied when:

- `host-orch` and `seeded-orch` exist as runtime execution models;
- both compose with a runtime loop under runtime ownership;
- both use the same target recipe and operator table for each example;
- both validate single-device and distributed CUDA+NVSHMEM runs;
- scheduler, dispatcher, EventTensor, platform, and backend remain separate
  components;
- docs and tests prove users do not program `scheduler.run`, operator-side
  EventTensor waits/signals, or top-level `build_run` configuration.
