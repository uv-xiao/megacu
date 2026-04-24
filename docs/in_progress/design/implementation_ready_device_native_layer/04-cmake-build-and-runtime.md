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
that by compiling a small native materializer for
`PROGRAM gemm_allreduce_program` or by instantiating C++ templates that
materialize target metadata during the native build. The exact mechanism is an
implementation detail, but it must stay inside the native build graph and must
not emit new C++/CUDA source as the normal lowering mechanism.

For the first implementation, choose the materializer executable path because it
is easiest to test:

1. CMake compiles a tiny host executable
   `megacu_materialize_cuda_nvshmem_gemm_allreduce`.
2. That executable links the user descriptor source and Megacu host lowering
   libraries.
3. At build time, it calls
   `megacu::detail::materialize_program<gemm_allreduce_program>()`.
4. It writes:
   - `cuda_nvshmem_gemm_allreduce.megacu.json` for inspection tests;
   - `cuda_nvshmem_gemm_allreduce.megacu.bin` for compact runtime metadata.
5. The orchestrate target embeds or links the binary metadata as data.

This is metadata generation, not source generation. The materializer must not
write `.cc`, `.cu`, `.cuh`, `.ptx`, or `.cubin` files for the program.

Initial CMake API shape:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce
  PROGRAM gemm_allreduce_program
  SOURCES gemm_allreduce_orchestrate.cc
  KERNELS gemm_allreduce_kernels.cu
  OPS
    gemm_tile_produce=gemm_tile_produce_kernel
    allreduce_tile_consume=allreduce_tile_consume_kernel
  COMPONENTS cuda_nvshmem_static
  BACKEND_ENVELOPE
    TEAM_SIZE 2
)
```

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
- a resource slot in the program has no matching parameter in the declared
  orchestrate ABI;
- the selected backend cannot provide a required event scope or primitive;
- the selected backend needs a fixed target envelope, such as `TEAM_SIZE`, and
  the orchestrate target does not provide it;
- a dispatcher or scheduler cannot consume the program's domain shape.

## Resolution Pipeline

The design has three resolution stages:

1. **Authoring**: users write typed tags and labels:
   `gemm_allreduce_program`, `output_tile_domain`,
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
  -> dispatch_plan
  -> schedule_plan
  -> kernel_plan + backend_plan
  -> target_metadata(.json/.bin)
  -> compiled orchestrate target
```

Owner by stage:

- `program_ir`: `include/megacu/detail/program_ir.h` and
  `src/program/program_builder.cc`;
- `dispatch_plan`: `src/dispatcher/tiled_compute_comm_dispatch.*`;
- `schedule_plan`: `src/scheduler/static_persistent.*`;
- `kernel_plan`: `src/lowering/persistent_stitch.*`;
- `backend_plan`: `src/backends/nvshmem/lowering.*`;
- `launch/runtime views`: `include/megacu/platform/cuda.h`,
  `include/megacu/backends/nvshmem.h`, and
  `docs/in_progress/design/implementation_ready_device_native_layer/11-distributed-launch-and-framework-integration.md`;
- metadata writer/reader: `src/target/metadata.*`.

## Parameterized Orchestrate Necessity

The public runtime boundary should be a parameterized compiled orchestrate
function, not `exec.bind(...)` plus `exec.run(...)`.

The split that is still necessary is not a public API split. The build graph can
know the program structure, but it cannot know every runtime pointer, buffer
size, team handle, or tile count. The compiled target therefore has internal
typed slots:

- resource slots for values such as workspace and event storage;
- extent slots for dynamic domain sizes such as matrix tile counts;
- platform launch slots, supplied by typed function parameters such as
  `megacu::cuda::launch_view`;
- backend handle slots, supplied by typed function parameters such as
  `megacu::nvshmem::team_view`.

The parameterized function fills those slots internally. It gives users one
ordinary C++ call:

```cpp
cuda_nvshmem_gemm_allreduce_orchestrate(workspace, events, launch, team, problem);
```

This is preferable to exposing the slots directly:

| Alternative | Why not primary |
| --- | --- |
| Public `exec.bind(...)` plus `exec.run(...)` | Exposes an implementation mechanism after the target is already compiled and makes the runtime surface look like a two-phase mini-runtime. |
| Generic `runtime_env` bag | Reintroduces stringly lookup and hides required resources from the C++ signature. |
| Positional argument array | Compact but brittle; materialized metadata and user code can disagree silently. |
| Rebuild for every shape | Defeats repeated-run use cases and makes dynamic tile/token counts expensive. |
| Let kernels receive all raw pointers/handles manually | Pushes lowering details into every kernel and prevents scheduler/backend inspection. |
| Megacu-owned allocator/event pool | Makes Megacu responsible for allocation policy and framework integration decisions. |

For `cuda_nvshmem_gemm_allreduce`:

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
gemm_ar_workspace workspace{a, b, partial, c, scratch};
megacu::event_storage_view events{event_buffer, event_bytes, true};
megacu::cuda::launch_view launch{stream, device_ordinal};
megacu::nvshmem::team_view team{nvshmem_team, my_pe, n_pes, world_pe, world_n_pes,
                                device_ordinal, megacu::nvshmem::ownership::external};
gemm_ar_problem problem{M, N, K, strides, tile_shape};

cuda_nvshmem_gemm_allreduce_orchestrate(workspace, events, launch, team, problem);
```

There is no primary runtime API for loading a module by path or passing an
untyped environment bag.

The direct function is the compiled target ABI:

```cpp
void cuda_nvshmem_gemm_allreduce_orchestrate(
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
4. return status or propagate platform/backend errors.

`run` must not:

- parse names;
- allocate the workspace;
- choose participant mappings;
- choose dispatcher/scheduler/lowering/backend;
- create CMake targets;
- compile or link code.

## What Changed From The Previous Model

The earlier redesign still looked too much like a plugin runtime, and then too
much like a generated-wrapper system.

The new design removes that ambiguity:

- CMake component targets own reusable engine compilation
- the user's compiled orchestrate program is the primary runtime surface
- `run` remains the repeated fast-path execution inside that program

There is no runtime phase that still secretly chooses scheduler or lowering.

## Dynamic Behavior

The lifecycle handles dynamic behavior at two public levels:

- **component-target structure**: which reusable dispatcher/scheduler/lowering
  engines exist
- **compiled orchestrate function**: how one concrete orchestrate program is
  compiled/linked against those engines, including which parameters remain
  dynamic at call time

Runtime dynamics are just parameters of the compiled orchestrate function. This
avoids turning `run` into a hidden graph builder or runtime compiler while also
avoiding a public bind/run layer after compilation.

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
