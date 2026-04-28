# Build, Link, And Configuration

CMake configures what implementation is linked. Runtime C++ runs that linked
implementation. There is no Megacu compiler step between those two facts.

## PR #4 Proof Boundary

PR #4 may link a tiny problem-specific proof target with direct CMake rules or
minimal helper functions if that keeps the architecture review focused. Those
rules are proof scaffolding only. The reusable CMake APIs below belong to the
future general concrete implementation in `docs/todo/concrete_impl/`.

## ConfigureTarget

The first configured component target remains useful:

```cmake
megacu_add_configure_target(
  NAME cuda_nvshmem_static
  DISPATCHER annotated_runtime
  SCHEDULER static_phased_or_overlap
  PLATFORM cuda
  BACKEND nvshmem)
```

But the meaning changes. This function links runtime component libraries. It
must not run a materializer or create metadata artifacts.

The dispatcher belongs to the `ConfigureTarget`, but it must be general to the
platform/backend/scheduler capability. It consumes virtual-participant
annotations supplied by each `OrchTarget` at runtime. It must not be named or
implemented as a GEMM+AllReduce-only dispatcher.

That is the general implementation target. PR #4's proof target may use
example-local mapping code if the code path still demonstrates normal
runtime-linked C++/CUDA execution and does not create generated metadata.

Expected linked implementation files:

```text
src/dispatcher/annotated_runtime.cc
src/scheduler/static_runtime.cc
src/platform/cuda/validation.cc
src/backends/nvshmem/runtime.cc
src/backends/nvshmem/validation.cc
src/target/runtime.cc
```

`src/lowering/` should disappear unless it is renamed to a runtime concept such
as `src/operators/` or `src/target/linked_symbols.cc`. If there is no lowering,
we should not keep a directory named lowering.

`megacu_add_configure_target` should fail if a requested component name is
unknown. It should not silently link a default placeholder.

## OrchTarget

Each full example variant links a direct target:

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
dispatcher: annotated_runtime
scheduler: overlap_gemm_ar
operator symbols: linked native CUDA/NVSHMEM functions
supported team sizes: 1 or 2 for the first slice
supported dtype/layout: f32 row-major first slice
```

This envelope can be exposed through target properties, a generated config
header, or linked constant data. Prefer linked constant data in handwritten
source for the first general implementation. It must be small and static. It
should not contain dispatch tables or schedule entries.

The `OrchTarget` supplies workload-specific facts that are not part of the
reusable `ConfigureTarget`:

- direct ABI and problem/workspace types;
- virtual participants and their attributes;
- native operator symbols;
- workload capability facts such as dtype/layout and supported problem shapes.

The `ConfigureTarget` supplies:

- platform and backend validation;
- the general annotated dispatcher algorithm;
- scheduler implementation;
- target-runtime helpers and common capability checks.

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

- dispatcher retargeting between single-card and two-card work based on
  `team.team_n_pes`;
- dispatcher tile iteration based on `problem.m`, `problem.n`, `tile_m`, and
  `tile_n`;
- dispatcher peer identity based on the current team;
- scheduler progress choices inside the linked scheduler envelope;
- backend fallback only when the linked backend explicitly advertises the
  optional primitive and the fallback has the same semantic contract.

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
