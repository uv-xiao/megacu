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

## Orchestrate-Program Targets

An authored orchestrate program should become a normal CMake target that reuses
those compiled artifacts.

What CMake/program compilation does:

- run the chosen dispatcher logic on the concrete program
- derive scheduler payloads and event/resource tables
- choose and parameterize the kernel-lowering engine
- compile/link the user-authored orchestrate code with the chosen components
- optionally emit inspection metadata if the selected backend needs it

The important boundary is that this step should reuse existing component
artifacts by default. It should not force recompilation of reusable engines.

## Runtime Surface

The normal runtime path should look like ordinary linked C++ code.

That means:

- no string path loading in ordinary user code
- no generic `runtime_env` bag in ordinary user code
- no extra generated runtime example when the authored orchestrate program
  itself can be compiled and called

The runtime should not re-decide strategy.

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
