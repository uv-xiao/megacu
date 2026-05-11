# Design: Example Ergonomics

## Goal

Make the CUDA+NVSHMEM examples show the intended Megacu authoring model:

- write small operator kernels;
- write one explicit task recipe;
- pick host-orch or seeded-orch;
- let Megacu own repeated arena, launch, operator-table, and wrapper
  scaffolding.

The examples should not teach users that every target must hand-write a large
`_arena.cuh`, `megacu.cu`, and `orchestrate.cc` stack. That stack was useful
while proving the architecture, but it is now hiding the simple idea that
Megacu composes small operator tasks into one runtime-owned mega-kernel.

## Current Example Shape

The active CUDA+NVSHMEM examples have the same repeated file pattern:

| Example | Heavy Megacu files | Lines |
| --- | --- | ---: |
| GEMM-AllReduce | `gemm_allreduce_arena.cuh`, `megacu_gemm_allreduce.cu`, `gemm_allreduce_orchestrate.cc` | 645 |
| GEMM-RS | `gemm_reduce_scatter_arena.cuh`, `gemm_reduce_scatter_megacu.cu`, `gemm_reduce_scatter_orchestrate.cc` | 588 |
| AG-GEMM | `allgather_gemm_arena.cuh`, `allgather_gemm_megacu.cu`, `allgather_gemm_orchestrate.cc` | 608 |
| Tiny decode | `tiny_decode_arena.cuh`, `tiny_decode_arena.cc`, `tiny_decode_megacu.cu`, `tiny_decode_runtime_recipe.cuh` | 583 |

Total code under `examples/cuda_nvshmem/*/megacu/` is about 2.4K lines. Some of
that is real target logic, but a large part is repeated runtime scaffolding.
The repeated shape is what makes the examples look heavy.

## What The Heavy Files Contain

### `_arena.cuh`

Current contents:

- compile-time task, EventTensor, dependency, and region capacities;
- host-side arrays of `device_task_record`, `device_event_tensor_record`,
  dependency refs, arena regions, completion flags, and remaining-work counts;
- `initial_remaining_work(...)` logic that re-derives work count from dispatch
  attrs;
- host-orch frame construction;
- copying sealed frame records into the host arena storage.

Why it exists:

- host-orch needs a concrete region that contains task records before the
  mega-kernel launch;
- seeded-orch needs the same capacity constants to reserve device-build space;
- the runtime loop needs completion and remaining-work side arrays;
- examples currently own fixed capacities because Megacu has no reusable arena
  storage helper.

What is target-specific:

- capacity numbers, if the target intentionally fixes them;
- the recipe type used to build the arena;
- target-specific maximum tile or event counts.

What is boilerplate:

- the storage struct layout;
- the `frame.seal()` copy loops;
- completion and remaining-work initialization;
- capacity-to-view conversion.

### `megacu.cu`

Current contents:

- device operator structs for each operator slot;
- target-specific operator dispatch table;
- seeded-orch recipe wrapper;
- `runtime::device_persistent<...>` type alias;
- EventTensor storage reset;
- NVSHMEM barrier setup for distributed paths;
- runtime object construction;
- CUDA `launch_megakernel(...)` call;
- exported C ABI entrypoints for host-orch and seeded-orch.

Why it exists:

- C++ templates for the runtime, loop, scheduler, dispatcher, EventTensor, and
  operator table must be linked into a concrete CUDA translation unit;
- operator bodies are device code and need direct access to target args;
- the first implementation wanted every runtime component visible in one file
  for review.

What is target-specific:

- operator bodies;
- mapping from `op_slot` to operator body;
- target runtime args;
- target event storage shape, such as tile count and team size;
- target-specific grid/block launch defaults, if they are intentionally chosen.

What is boilerplate:

- `cuda_status(...)`;
- `require_stream(...)`;
- `runtime_event_count(...)`;
- seeded recipe wrapper;
- `device_persistent` type alias shape;
- EventTensor zeroing and distributed pre-launch barrier;
- host-orch and seeded-orch exported entrypoint bodies.

### `orchestrate.cc`

Current contents:

- `cudaMallocManaged` helper;
- managed arena allocation and release;
- host arena to device arena copy;
- seeded arena clearing;
- device status and construction status allocation;
- stream/device validation;
- public host-callable wrappers around the C ABI runtime entrypoints.

Why it exists:

- examples need a host-facing function that accepts raw CUDA-like target args;
- host-orch needs the host-built records copied into launch-visible storage;
- seeded-orch needs empty arena storage plus status cells visible to device and
  host;
- managed allocation is a simple way to make the smoke examples work on one or
  two GPUs without introducing a separate allocator API.

What is target-specific:

- public function names;
- target args passed through to runtime args;
- target-specific arena capacities;
- any target-specific host precondition.

What is boilerplate:

- managed allocation lifecycle;
- arena copy and clear loops;
- device status allocation and synchronization pattern;
- duplicate host-orch and seeded-orch wrapper structure.

## Why The Examples Became Heavy

The heaviness is not because Megacu requires a high-level graph or complicated
operator API. It comes from missing helper ownership after the architecture was
reset.

The reset deliberately made the architecture explicit:

```text
host orchestrate
  -> build or seed task arena
  -> launch one mega-kernel

device runtime loop
  -> scheduler says which task is ready
  -> dispatcher maps task attrs to block work
  -> EventTensor completes sync-only tasks
  -> operator table invokes small kernels
```

During proof work, each example spelled out every layer. That made correctness
review possible, but it left all of the following in the example directories:

- arena materialization;
- execution model bridge;
- CUDA launch bridge;
- EventTensor storage reset bridge;
- operator-table adapter bridge.

Those are real requirements of the runtime architecture, but they are not all
requirements for a target author. A target author should mostly provide:

- raw target ABI;
- task recipe;
- operator slot enum;
- operator bodies;
- problem-specific golden/baseline validation.

## Optimization Space

### 1. Consolidate Arena Storage

Move generic host/device arena storage into Megacu, for example:

```cpp
using arena = megacu::runtime::fixed_arena<
    TaskCapacity, EventCapacity, DepCapacity, RegionCapacity>;

arena host;
auto status = megacu::runtime::build_host_arena(host, args, build_recipe);
```

Potential benefit:

- removes most `_arena.cuh` duplication;
- keeps capacities explicit;
- keeps user responsible for choosing enough capacity;
- makes host-orch and seeded-orch use the same capacity object.

Risks:

- if capacity policy becomes implicit, users may think Megacu catches missing
  capacity or dependency mistakes;
- helper must remain a storage helper, not a graph builder.

### 2. Consolidate Managed Launch Arena

Move `cudaMallocManaged` arena allocation, copy, clear, and release into a CUDA
platform helper:

```cpp
auto arena = megacu::platform::cuda::managed_task_arena::allocate(capacity);
arena.copy_from(host_arena);
arena.clear_for_seeded();
```

Potential benefit:

- removes repeated `managed_arena` structs from every `orchestrate.cc`;
- makes examples show raw args and runtime selection instead of memory plumbing.

Risks:

- managed memory is a convenience path, not necessarily the production memory
  strategy. The helper name must make that explicit.

### 3. Consolidate Host-Orch And Seeded-Orch Wrappers

Provide a helper that turns a target runtime descriptor into two host-callable
entrypoints or implementation functions:

```cpp
return megacu::runtime::cuda_nvshmem::launch_host_orch(
    target_descriptor{driver, args, operators, recipe});

return megacu::runtime::cuda_nvshmem::launch_seeded_orch(
    target_descriptor{driver, args, operators, recipe});
```

Potential benefit:

- removes duplicated wrapper shape from GEMM-AllReduce, GEMM-RS, and AG-GEMM;
- keeps runtime model choice explicit;
- centralizes status propagation and stream synchronization.

Risks:

- too much descriptor machinery can become a hidden framework. The descriptor
  should contain only already-explicit pieces: driver, args, capacities, recipe,
  operator table, and launch dimensions.

### 4. Consolidate Runtime Launch Glue

Provide a small reusable CUDA runtime launcher:

```cpp
using runtime = megacu::runtime::cuda_nvshmem::block_tile_runtime<
    ExecutionModel, Operators>;

return runtime::launch(stream, launch_shape, arena, event_tensor, operators);
```

Potential benefit:

- removes repeated `device_persistent` aliases and `launch_megakernel` calls;
- keeps scheduler, dispatcher, loop, and EventTensor selection visible through
  the type name or descriptor;
- makes operator kernels the largest part of `megacu.cu`, which is the right
  emphasis.

Risks:

- this helper must not force one scheduler or runtime loop forever. It should
  be a selected strategy, not the only runtime API.

### 5. Improve Operator Authoring API

Keep operator kernels as plain device callables, but give them a common thin
context and optional helpers for common tile math:

```cpp
struct gemm_tile {
  template <class Ctx>
  __device__ void operator()(Ctx ctx, tile_work tile) const;
};
```

Potential benefit:

- reduces hand-written tile id decoding repeated across examples;
- keeps backend/platform intrinsics out of operators except where the operator
  genuinely does communication;
- keeps task granularity at operator/task level, not EventTensor subtasks.

Risks:

- helper APIs must not hide communication or synchronization. EventTensor
  notify/wait remains task attrs, not operator-side protocol.

## Recommended Direction

Use a two-layer solution.

Layer 1: Megacu runtime authoring helpers

- `runtime::fixed_arena` and `runtime::capacity`;
- host arena build/copy helpers;
- seeded arena reset/status helpers;
- reusable host-orch and seeded-orch launch functions;
- reusable CUDA+NVSHMEM block-tile runtime descriptor.

Layer 2: tiny example-facing target descriptors

Each example keeps a small descriptor that says:

```cpp
struct target {
  using args = runtime_args;
  using operators = gemm_rs_operators;
  static constexpr auto capacity = megacu::runtime::capacity{...};
  static constexpr auto launch = megacu::platform::cuda::launch_shape{...};

  template <class Orch>
  MEGACU_HOST_DEVICE static megacu::status build(Orch &orch, args value) {
    return build_runtime_recipe(orch, value);
  }
};
```

Then the example implementation becomes mostly:

```cpp
extern "C" megacu::status cuda_nvshmem_gemm_rs_host_orch(...raw args...) {
  return megacu::runtime::cuda_nvshmem::host_orch<target>(
      driver, target::args{...});
}

extern "C" megacu::status cuda_nvshmem_gemm_rs_seeded_orch(...raw args...) {
  return megacu::runtime::cuda_nvshmem::seeded_orch<target>(
      driver, target::args{...});
}
```

The exact names are not frozen by this draft. The ownership is the important
part: target descriptors describe target facts; Megacu helpers perform generic
runtime mechanics.

## What Must Not Change

- No tensor input/output/inout argument abstraction.
- No dependency inference from pointer use.
- No hidden EventTensor subtasks inside one operator task.
- No fallback checking for missed dependencies, bad capacities, or bad user
  buffers.
- No removal of golden/baseline correctness paths.
- No single hard-coded runtime loop that prevents future runtime strategies.

## Examples

### GEMM-RS Before

The reader must inspect:

- `common/gemm_reduce_scatter_runtime_recipe.cuh` for task graph shape;
- `megacu/gemm_reduce_scatter_arena.cuh` for host arena storage;
- `megacu/gemm_reduce_scatter_megacu.cu` for operators and runtime launch;
- `megacu/gemm_reduce_scatter_orchestrate.cc` for host wrapper allocation.

The algorithm is small, but the path to the algorithm is long.

### GEMM-RS After

The reader should inspect:

- `common/gemm_reduce_scatter_runtime_recipe.cuh` for task graph shape;
- `megacu/gemm_reduce_scatter_operators.cuh` or equivalent for operator
  bodies;
- one small target descriptor/wrapper for raw API binding.

Generic arena and launch mechanics should live under `include/megacu/`.

### Tiny Decode

Tiny decode is a useful stress case because it is not a distributed GEMM
collective. If the helper design only works for GEMM-RS and AG-GEMM, it is too
specific. The same helper must support local CUDA EventTensor storage and
operator pipelines.

## Contracts

- Operator kernels use a common thin context and do not call scheduler APIs.
- Recipes submit tasks, dependencies, and EventTensor attrs explicitly.
- Runtime wrappers are Megacu-owned, not copied per example.
- Capacity remains explicit and user-owned.
- Examples may still own golden/baseline code and workload-specific
  validation.
- No heavy validation or fallback checking is added.
- Helpers expose selected runtime strategy; they do not erase host-orch versus
  seeded-orch.

## Failure Modes

- If helpers hide too much, examples stop showing the architecture.
- If helpers stay under `examples/`, Megacu does not actually become easier to
  use outside examples.
- If capacity policy becomes implicit, users may assume Megacu repairs missing
  task/dependency capacity.
- If runtime launch helpers are too specialized, they freeze the current
  block-tile loop and make future strategies harder.
- If operator helpers include synchronization, EventTensor ownership leaks back
  into operators.

## Verification

Required before merge:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
MEGACU_DOCKER_GPUS='"device=0,1"' \
MEGACU_NVSHMEM_IMAGE='megacu-nvshmem:cuda12.8' \
MEGACU_NVSHMEM_BUILD_DIR='build-nvshmem' \
tools/cuda_nvshmem/run_two_card_docker.sh
```

Additional PR-specific checks:

- static check that examples no longer define per-example managed arena helper
  structs when a Megacu-owned helper exists;
- static check that examples do not call scheduler APIs from operators;
- static check that example operator files do not call manual EventTensor task
  wait/notify helpers;
- line-count comparison for the four Megacu example folders before and after
  the refactor;
- tutorial review confirming each example explains recipe and operators before
  runtime scaffolding.

## Out Of Scope

- A high-level tensor graph builder.
- Automatic dependency inference from tensor access.
- Runtime fallback checking for missed dependencies or invalid user buffers.
- Benchmark claims.
- Removing baseline/golden implementations.

## Closeout

When the PR implementation is accepted:

- promote the accepted helper API and example authoring rules into
  `docs/design/`;
- update example tutorials to show the smaller authoring path;
- remove this in-progress design draft before merging;
- keep future usability or generator work under `docs/todo/`.
