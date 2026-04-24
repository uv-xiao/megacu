# Language Responsibilities

This chapter answers the implementation question directly: what should be done
in authoring C++, what should be done in CMake/native build rules, and what
should be done in runtime C++/CUDA?

## Authoring C++ Layer

Authoring C++ is the right place for:

- authoring the orchestrate program
- defining explicit runtime argument types
- registering named kernels and native adapters
- expressing the orchestration through a small builder surface

This authoring code is the thing that gets compiled and run.

## CMake And Native Build Layer

CMake/native build rules are a separate responsibility from both authoring C++
and runtime C++.

They are the right place for:

- compiling reusable dispatcher/scheduler/kernel-lowering targets
- compiling reusable kernels and helper code
- compiling/linking the authored orchestrate program target
- materializing target metadata and selecting the required low-level
  implementations
- linking platform/backend adapters
- producing optional inspection metadata

In other words, Megacu should separate:

1. reusable compilation
2. later orchestrate-program compilation/linking

Both are offline native build-graph steps, and both stay out of the runtime
API.

## C++ And CUDA Layer

C++/CUDA is the right place for:

- named kernel implementations
- backend primitive headers and device calls
- platform adapters
- backend adapters
- launch adapters for process-model bootstrap and typed runtime-view
  construction
- reusable lowering implementations and linked target metadata
- the authored orchestrate program
- the internal `run(...)` path

This keeps the hot path close to native code and lets the orchestrate program
be a regular compiled target.

## Runtime C++ API

The runtime C++ API should be small:

- call the compiled orchestrate program
- construct typed launch/backend views through optional launch adapters when the
  caller does not already own them

It should not:

- expose CMake/build steps
- expose strategy selection
- expose dispatcher, scheduler, lowering, platform, or backend plan objects
- expose packaging concerns that belong to the offline build path
- initialize or finalize NVSHMEM inside the compiled orchestrate target

That is the core boundary for implementation.

## First Implementation Recommendation

For the first implementation, the cleanest split is:

- **CMake/native build rules**
  - reusable component-target hooks
  - orchestrate-target hooks
  - target metadata materialization
  - selection/linking of required backend-provided low-level implementations
  - CUDA/C++ compilation and linking

- **C++/CUDA**
  - orchestrate-program authoring
  - kernels
  - reusable dispatcher/scheduler/lowering/platform/backend implementations
  - optional MPI/NVSHMEM launch adapter
  - `run(...)`

- **Framework integration**
  - optional Torch Distributed adapter code outside Megacu core that reads
    rank/world state, exchanges the NVSHMEM UID, owns symmetric allocation
    wrappers, and calls the compiled C++ target through an extension binding
  - no framework types in `program_ir`, dispatcher, scheduler, kernel, or
    backend metadata-section records
  - no Python dependency in public program headers, materializers, metadata,
    lowering, backend adapters, or compiled target ABI
