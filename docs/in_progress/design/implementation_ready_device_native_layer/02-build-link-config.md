# Build, Link, And Configuration

CMake configures what implementation is linked. Runtime C++ runs that linked
implementation. There is no Megacu compiler step between those two facts.

## Component Target

The first component target remains useful:

```cmake
megacu_add_components(
  NAME cuda_nvshmem_static
  DISPATCHER cuda_nvshmem_gemm_ar
  SCHEDULER static_phased_or_overlap
  PLATFORM cuda
  BACKEND nvshmem)
```

But the meaning changes. This function links runtime component libraries. It
must not run a materializer or create metadata artifacts.

Expected linked implementation files:

```text
src/dispatcher/cuda_nvshmem_gemm_ar_runtime.cc
src/scheduler/gemm_ar_runtime.cc
src/platform/cuda/validation.cc
src/backends/nvshmem/runtime.cc
src/backends/nvshmem/validation.cc
src/target/runtime.cc
```

`src/lowering/` should disappear unless it is renamed to a runtime concept such
as `src/operators/` or `src/target/linked_symbols.cc`. If there is no lowering,
we should not keep a directory named lowering.

`megacu_add_components` should fail if a requested component name is unknown.
It should not silently link a default placeholder.

## Orchestrate Target

Each example variant links a direct target:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_overlap
  COMPONENTS cuda_nvshmem_static
  SOURCES gemm_allreduce_overlap_orchestrate.cc
  OPERATORS megacu_cuda_gemm_allreduce_overlap_f32
  CAPABILITY cuda_nvshmem_gemm_allreduce_overlap_capability)
```

The CMake target records a capability envelope, not a materialized program:

```text
platform: cuda
backend: nvshmem
dispatcher: cuda_nvshmem_gemm_ar
scheduler: overlap_gemm_ar
operator symbols: linked native CUDA/NVSHMEM functions
supported team sizes: 1 or 2 for the first slice
supported dtype/layout: f32 row-major first slice
```

This envelope can be exposed through target properties, a generated config
header, or linked constant data. Prefer linked constant data in handwritten
source for the first implementation. It must be small and static. It should not
contain dispatch tables or schedule entries.

If a generated config header is ever considered, it needs a separate design
review because the current requirement is to avoid code generation as the
normal implementation mechanism.

## What Compilation Means

Megacu compilation means:

- compile Megacu runtime components;
- compile user/example orchestrate code;
- compile handwritten CUDA/NVSHMEM kernels;
- link the selected components and operator symbols into one target;
- expose a direct function ABI.

Megacu compilation does not mean:

- translating a C++ description into an IR;
- lowering IR into dispatch/schedule/backend sections;
- generating CUDA or C++ source;
- generating per-program binary metadata;
- creating a host materializer executable.

## Runtime Selection Boundary

Runtime may choose values inside the linked envelope:

- single-card versus two-card path based on `team.team_n_pes`;
- tile iteration based on `problem.m`, `problem.n`, `tile_m`, `tile_n`;
- peer rank based on the current team;
- fallback path when an optional backend primitive is unavailable.

Runtime must not choose a different linked component family:

- no switching from NVSHMEM to MPI collectives inside this target;
- no switching from phased to overlap scheduler unless the target is explicitly
  designed as a runtime-polymorphic target;
- no loading kernels by string or path.
- no choosing scheduler/backend through a runtime registry.
- no dispatch by opaque string names.

## Symbol And Capability Checks

The CMake layer should make linkage failures obvious:

```text
orchestrate target
  requires target capability symbol
  requires scheduler runtime symbol
  requires dispatcher runtime symbol
  requires native operator symbols
  links platform/backend validation and primitive helpers
```

The first implementation can prove this with compile/link tests and direct
symbol references. It does not need a materialized metadata file.

## CMake Failure Modes

CMake should reject:

- missing required operator symbols;
- unsupported platform/backend pair;
- unsupported scheduler/backend combination;
- example target without its own `CMakeLists.txt`;
- per-example Docker/tool duplication unless justified by the example.

Runtime should reject:

- invalid CUDA stream/device;
- invalid NVSHMEM team;
- unsupported team size;
- storage not symmetric for multi-card communication;
- unsupported dtype/layout/problem shape for the linked target.
