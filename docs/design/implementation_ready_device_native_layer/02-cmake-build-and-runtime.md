# CMake Build And Runtime

This chapter defines the build/execute flow.

The key rule is:

> CMake owns the offline build graph; runtime C++ runs the compiled orchestrate
> program directly.

## Build Graph

The primary Megacu story should be CMake-managed.

The build graph may have two internal layers:

1. **reusable component targets**
2. **orchestrate-program targets**

The second layer depends on the first, so expensive reusable compilation can be
cached and reused without exposing extra lifecycle terms in the runtime API.

## Reusable Component Targets

CMake should compile reusable targets for:

- dispatcher implementations
- scheduler implementations
- kernel-lowering implementations
- platform/backend adapters
- reusable kernels and helper code

These are ordinary native build artifacts. They are not runtime API concepts.

Initial CMake API shape:

```cmake
megacu_add_components(
  NAME cuda_nvshmem_static
  DISPATCHER tile_dispatch
  SCHEDULER static_persistent
  KERNEL_LOWERING persistent_stitch
  PLATFORM cuda
  BACKEND nvshmem
)
```

The implementation owner should be `cmake/MegacuTargets.cmake`. Component
targets should link implementation code from `src/dispatcher/`,
`src/scheduler/`, `src/lowering/`, `src/platform/cuda/`, and
`src/backends/nvshmem/`.

Concrete first implementation:

- `megacu_components_cuda_nvshmem_static_dispatcher`: object or static library
  built from `src/dispatcher/tiled_compute_comm_dispatch.*`.
- `megacu_components_cuda_nvshmem_static_scheduler`: object or static library
  built from `src/scheduler/static_persistent.*`.
- `megacu_components_cuda_nvshmem_static_lowering`: object or static library
  built from `src/lowering/persistent_stitch.*`.
- `megacu_components_cuda_nvshmem_static_cuda`: object or static library built
  from `src/platform/cuda/*`.
- `megacu_components_cuda_nvshmem_static_nvshmem`: object or static library
  built from `src/backends/nvshmem/*`.
- `cuda_nvshmem_static`: interface target linking the five component artifacts.

The exact target names may be normalized by CMake, but build output must expose
the same decomposition so tests can prove component reuse.

## Orchestrate-Program Targets

An authored orchestrate program should become a normal CMake target that reuses
those compiled artifacts.

## What Megacu Compilation Means

Megacu compilation is ordinary native build work:

- compile user-authored C++ orchestrate descriptor and wrapper sources;
- compile user-authored CUDA/C++ kernel sources;
- compile reusable Megacu dispatcher, scheduler, kernel-lowering, platform, and
  backend implementation targets;
- link the orchestrate target against the selected reusable implementations;
- materialize compact target metadata such as op slots, resource slots, event
  slots, dispatch tables, schedule payloads, participant tables, and backend
  layouts;
- optionally write inspection metadata for humans and tests.

Megacu compilation must not:

- translate the orchestrate descriptor into new CUDA source;
- translate kernels into another programming language or DSL;
- emit per-program C++/CUDA source as the normal lowering mechanism;
- generate a new runtime wrapper that becomes the primary public API;
- run CMake/build logic from runtime C++;
- choose dispatcher, scheduler, kernel lowering, platform, or backend at
  runtime.

The key implementation rule is: **target lowering selects and links existing
low-level implementations provided by Megacu/platform/backend components, then
materializes data needed by those implementations. It does not synthesize new
kernel code.**

What CMake/program compilation does:

- run the chosen dispatcher logic on the concrete program
- derive scheduler payloads and event/resource tables
- assign compact slots to op, domain, participant, resource, and event tags
- choose and parameterize the kernel-lowering implementation
- link the required dispatcher, scheduler, lowering, platform, backend, and
  kernel implementations
- compile/link the user-authored orchestrate code with the chosen components
- optionally write inspection metadata if the selected backend needs it

The important boundary is that this step should reuse existing component
artifacts by default. It should not force recompilation of reusable engines.

The first implementation should obtain program metadata from the explicit C++
descriptor, not from a Python tool and not from runtime parsing. CMake can do
that by compiling small native materializers for
`PROGRAM gemm_allreduce_phased_program` and
`PROGRAM gemm_allreduce_overlap_program`, or by instantiating C++ templates that
materialize target metadata during the native build. The exact mechanism is an
implementation detail, but it must stay inside the native build graph and must
not emit new C++/CUDA source as the normal lowering mechanism.

For the first implementation, choose the materializer executable path because it
is easiest to test:

1. CMake compiles tiny host executables
   `megacu_materialize_cuda_nvshmem_gemm_allreduce_phased` and
   `megacu_materialize_cuda_nvshmem_gemm_allreduce_overlap`.
2. That executable links the user descriptor source and Megacu host lowering
   libraries.
3. At build time, each executable calls the matching
   `megacu::detail::materialize_program<...>()`.
4. They write:
   - `cuda_nvshmem_gemm_allreduce_phased.megacu.json` and
     `cuda_nvshmem_gemm_allreduce_overlap.megacu.json` for inspection tests;
   - `cuda_nvshmem_gemm_allreduce_phased.megacu.bin` and
     `cuda_nvshmem_gemm_allreduce_overlap.megacu.bin` for compact runtime
     metadata.
5. CMake turns each `.megacu.bin` into a linked binary-object target or
   equivalent native data section.
6. The orchestrate target links that metadata object and accesses it through
   `megacu::detail::linked_target_metadata(...)`.

This is metadata generation, not source generation. The materializer must not
write `.cc`, `.cu`, `.cuh`, `.ptx`, or `.cubin` files for the program.
The linked binary object must not contain raw C++ pointers, spans, string views,
type indexes, or allocator-owned state; see
`10-implementation-architecture.md`.

Initial CMake API shape:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_phased
  PROGRAM gemm_allreduce_phased_program
  SOURCES gemm_allreduce_phased_orchestrate.cc
  KERNELS gemm_allreduce_kernels.cu
  OPS
    gemm_tile_produce
      LAUNCH gemm_tile_produce_kernel
    allreduce_tile_consume
      LAUNCH allreduce_tile_consume_kernel
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE phased
  BACKEND_ENVELOPE
    TEAM_SIZE 2
)

megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_overlap
  PROGRAM gemm_allreduce_overlap_program
  SOURCES gemm_allreduce_overlap_orchestrate.cc
  KERNELS gemm_allreduce_kernels.cu
  OPS
    gemm_tile_produce
      LAUNCH gemm_tile_produce_kernel
      CALLABLE gemm_tile_produce_body
    allreduce_tile_consume
      LAUNCH allreduce_tile_consume_kernel
      CALLABLE allreduce_tile_consume_body
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE co_resident_persistent
  BACKEND_ENVELOPE
    TEAM_SIZE 2
)
```

The phased target above only needs `LAUNCH` entrypoints because it can run
ordinary CUDA kernels. The overlap target lists `CALLABLE` entrypoints because
`persistent_stitch` may call op bodies from inside a persistent kernel. A
different overlap lowering that launches co-resident kernels directly may omit
`CALLABLE`.

Required properties:

- component targets are reusable build artifacts;
- orchestrate targets depend on component targets;
- strategy choice happens in CMake/native build metadata;
- CMake selects or links the dispatcher component, but the dispatcher owns
  virtual participant placement and emits participant mapping metadata;
- runtime C++ cannot choose a different dispatcher, scheduler, lowering,
  platform, or backend for that target;
- the build can write inspection metadata for selected linked implementations
  and materialized target metadata.

The first implementation should fail CMake configure or build if:

- an `OPS` key in CMake has no matching op tag in `program_ir`;
- a program op has no implementation symbol in `OPS`;
- the selected lowering mode requires an entrypoint role, such as `LAUNCH`,
  `CALLABLE`, or a lowering-provided trampoline, that the `OPS` entry and
  lowering component do not provide;
- a resource slot in the program has no matching parameter in the declared
  orchestrate ABI;
- the selected backend cannot provide a required event scope or primitive;
- the selected backend needs a fixed target envelope, such as `TEAM_SIZE`, and
  the orchestrate target does not provide it;
- a dispatcher or scheduler cannot consume the program's domain shape.

## Resolution Pipeline

The design has three resolution stages:

1. **Authoring**: users write typed tags and labels:
   `gemm_allreduce_phased_program`, `gemm_allreduce_overlap_program`,
   `output_tile_domain`,
   `partial_ready_event`, `compute_lane`, `"partial_ready"`.
2. **Target lowering**: CMake-selected components turn tags into compact target
   metadata:
   domain slot, event slot, participant slot, dispatch table, schedule payload,
   backend event layout. This is data materialization and linking, not source
   generation.
3. **Parameterized call**: the compiled orchestrate function receives typed
   runtime values:
   workspace views, event storage, CUDA launch view, NVSHMEM team, and a problem
   descriptor inside the target envelope. The function implementation fills the
   already-lowered slots internally before launching the selected execution path.

Labels never drive runtime lookup. They appear in diagnostics, metadata dumps,
and verification output.

Concrete metadata pipeline:

```text
program_builder
  -> program_ir
  -> target metadata model
       program section
       dispatch section
       schedule section
       kernel section
       backend section
  -> target_metadata(.json/.bin)
  -> linked metadata object
  -> compiled orchestrate target
```

These sections are not public plan APIs. They are private materializer outputs
inside one target metadata model. Component names may still appear in internal
C++ record names while the first implementation is being built, but runtime
C++ consumes the linked metadata blob and does not see standalone plan objects.

Owner by section:

- `program_ir`: `include/megacu/detail/program_ir.h` and
  `src/program/program_builder.cc`;
- dispatch section: `src/dispatcher/tiled_compute_comm_dispatch.*`;
- schedule section: `src/scheduler/static_persistent.*`;
- kernel section: `src/lowering/persistent_stitch.*`;
- backend section: `src/backends/nvshmem/lowering.*`;
- `launch/runtime views`: `include/megacu/platform/cuda.h`,
  `include/megacu/backends/nvshmem.h`, and
  `docs/design/implementation_ready_device_native_layer/09-distributed-launch-and-framework-integration.md`;
- metadata writer/reader: `src/target/metadata.*`.
- metadata ABI and embedding guardrails:
  `docs/design/implementation_ready_device_native_layer/10-implementation-architecture.md`.

## Parameterized Orchestrate Slot Fill

The compiled target has internal typed slots for runtime values the build graph
cannot know:

- resource slots for values such as workspace and event storage;
- extent slots for dynamic domain sizes such as matrix tile counts;
- platform launch slots, supplied by typed function parameters such as
  `megacu::cuda::launch_view`;
- backend handle slots, supplied by typed function parameters such as
  `megacu::nvshmem::team_view`.

The parameterized function fills those slots internally. It gives users one
ordinary C++ call:

```cpp
auto status = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    workspace, events, launch, team, problem);
```

For `cuda_nvshmem_gemm_allreduce_overlap`:

- `workspace` fills tensor, partial-output, final-output, and scratch storage
  slots;
- `events` fills the synchronization storage slot used by
  `partial_ready_event`;
- `launch` fills the CUDA stream/device slot;
- `team` fills the backend handle slot;
- `problem` fills the runtime matrix shape, stride, datatype envelope, and tile
  extents, causing the corresponding logical output tiles to run under the
  already chosen dispatcher/scheduler/lowering/backend.

## Runtime Surface

The normal runtime path should look like ordinary linked C++ code.

That means:

- no string path loading in ordinary user code
- no generic `runtime_env` bag in ordinary user code
- no extra generated runtime example when the authored orchestrate program
  itself can be compiled and called

The runtime should not re-decide strategy.

Runtime code should link the orchestrate target and call an explicit function:

```cpp
megacu::symmetric_buffer_view partial_buffer =
    session.symmetric_alloc(partial_bytes, 128);
gemm_ar_workspace workspace{a, b, typed_symmetric(partial_buffer, dtype), c,
                            scratch};
megacu::symmetric_buffer_view event_buffer =
    session.symmetric_alloc(event_bytes, alignof(std::uint64_t));
megacu::event_storage_view events{event_buffer};
megacu::cuda::launch_view launch{stream, device_ordinal};
megacu::nvshmem::team_view team{nvshmem_team, my_pe, n_pes, world_pe, world_n_pes,
                                device_ordinal, backend, session,
                                megacu::nvshmem::ownership::external};
gemm_ar_problem problem{M, N, K, strides, tile_shape};

auto status = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    workspace, events, launch, team, problem);
```

There is no primary runtime API for loading a module by path or passing an
untyped environment bag.

The direct function is the compiled target ABI:

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
```

## Orchestrate

`orchestrate` is the authored orchestration program itself.

It should accept concrete, explicit typed arguments, not a generic environment
object.

It may:

- accept the concrete buffers/views required by the program
- accept the concrete CUDA stream/device view required by the program
- accept the concrete communicator/backend handles required by the program
- allocate or attach runtime-owned state required by that program
- validate max envelopes that were left runtime-configurable
- call `run(...)` internally for the current invocation

It must not:

- choose a different dispatcher
- choose a different scheduler
- choose a different kernel lowering
- trigger CMake/build work
- rebuild the target

Those decisions were already made by the build graph.

## Run

`run` is the internal repeated fast-path device execution inside the compiled
orchestrate target. It is not a separate public object users call after
constructing the compiled target.

If a structural property changes, the target may need to be rebuilt by the
build graph. `run` must not absorb that work.

For the first static persistent CUDA/NVSHMEM target, `run` should do only:

1. validate runtime extents against the target envelope;
2. map explicit runtime views and backend handles to lowered slots;
3. launch the selected persistent CUDA path or direct launch sequence;
4. return `megacu::status` for validation, launch, or backend errors.

`run` must not:

- parse names;
- allocate the workspace;
- choose participant mappings;
- choose dispatcher/scheduler/lowering/backend;
- create CMake targets;
- compile or link code.

## Dynamic Behavior

Runtime dynamics are parameters of the compiled orchestrate function. Target
structure is fixed by the build graph; `run` must not become a graph builder or
runtime compiler.

## Working Path

The intended working path is:

1. author kernels and native adapters in C++/CUDA
2. author the orchestrate program in C++
3. use CMake to build reusable component targets
4. use CMake to compile/link the orchestrate target against them
5. run the compiled program

## Build And Runtime Evidence

The first implementation must include:

- a configure/build test proving `megacu_add_components(...)` creates a
  reusable target;
- a configure/build test proving `megacu_add_orchestrate_target(...)` links
  against that reusable target;
- a direct-call smoke test or compile check proving deployment code does not
  invoke CMake, build, or strategy-selection APIs at runtime.
