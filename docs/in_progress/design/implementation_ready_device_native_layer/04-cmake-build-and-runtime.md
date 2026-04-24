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

## Orchestrate-Program Targets

An authored orchestrate program should become a normal CMake target that reuses
those compiled artifacts.

What CMake/program compilation does:

- run the chosen dispatcher logic on the concrete program
- derive scheduler payloads and event/resource tables
- assign compact slots to op, domain, participant, resource, and event tags
- choose and parameterize the kernel-lowering engine
- compile/link the user-authored orchestrate code with the chosen components
- optionally emit inspection metadata if the selected backend needs it

The important boundary is that this step should reuse existing component
artifacts by default. It should not force recompilation of reusable engines.

The first implementation should obtain program metadata from the explicit C++
descriptor, not from a Python tool and not from runtime parsing. CMake can do
that by compiling a small native materializer for `PROGRAM event_copy_program`
or by instantiating C++ templates that emit target metadata during the native
build. The exact mechanism is an implementation detail, but it must stay inside
the native build graph and produce ordinary generated headers/objects.

Initial CMake API shape:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_event_copy
  PROGRAM event_copy_program
  SOURCES event_copy_orchestrate.cc
  KERNELS event_copy_kernels.cu
  OPS
    write_then_signal=write_then_signal_kernel
    wait_then_check=wait_then_check_kernel
  COMPONENTS cuda_nvshmem_static
  MAP producer_lane TO RANK 0
  MAP consumer_lane TO RANK 1
)
```

Required properties:

- component targets are reusable build artifacts;
- orchestrate targets depend on component targets;
- strategy choice happens in CMake/native build metadata;
- virtual participant mappings are resolved into target metadata;
- runtime C++ cannot choose a different dispatcher, scheduler, lowering,
  platform, or backend for that target;
- the build can emit inspection metadata for generated or lowered code.

## Resolution Pipeline

The design has three resolution stages:

1. **Authoring**: users write typed tags and labels:
   `event_copy_program`, `tile_domain`, `ready_event`, `producer_lane`,
   `"ready"`.
2. **Target lowering**: CMake-selected components turn tags into compact target
   metadata:
   domain slot, event slot, participant slot, dispatch table, schedule payload,
   backend event layout.
3. **Runtime binding**: the compiled orchestrate function receives typed
   runtime values:
   workspace views, event storage, NVSHMEM team, and dynamic extents inside the
   target envelope. These values fill the already-lowered slots.

Labels never drive runtime lookup. They appear in diagnostics, metadata dumps,
and verification output.

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
event_copy_workspace workspace{scratch, payload};
megacu::event_storage_view events{event_buffer, event_bytes};
megacu::nvshmem_team_view team{nvshmem_team};

cuda_nvshmem_event_copy_orchestrate(workspace, events, team, tiles);
```

There is no primary runtime API for loading a module by path or passing an
untyped environment bag.

The direct function can be a small wrapper around the compiled executor:

```cpp
void cuda_nvshmem_event_copy_orchestrate(
    event_copy_workspace workspace,
    megacu::event_storage_view events,
    megacu::nvshmem_team_view team,
    std::int32_t tiles) {
  megacu::executor<event_copy_program> exec{team};
  exec.bind<workspace_slot>(workspace);
  exec.bind<event_storage_slot>(events);
  exec.run(megacu::extent<tiles_extent>(tiles));
}
```

## Orchestrate

`orchestrate` is the authored orchestration program itself.

It should accept concrete, explicit typed arguments, not a generic environment
object.

It may:

- accept the concrete buffers/views required by the program
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

`run` is still the repeated fast-path device execution inside the orchestrate
program.

If a structural property changes, the target may need to be rebuilt by the
build graph. `run` must not absorb that work.

For the first static persistent CUDA/NVSHMEM target, `run` should do only:

1. validate runtime extents against the target envelope;
2. bind explicit runtime views and backend handles to lowered slots;
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

The lifecycle handles dynamic behavior at three different levels:

- **component-target structure**: which reusable dispatcher/scheduler/lowering
  engines exist
- **orchestrate-target structure**: how one concrete orchestrate program is
  compiled/linked against those engines
- **run-time dynamics**: extents, slices, scalars, and other values inside the
  program's validated envelope

This avoids turning `run` into a hidden graph builder or runtime compiler.

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
