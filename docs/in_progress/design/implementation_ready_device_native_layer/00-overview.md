# Implementation-Ready Megacu Device-Native Layer Design

This directory holds the active implementation-ready Megacu device-native
design in one flat, ordered set of files.

The picked stable direction remains in `docs/design/megacu_cpp_cuda_layer.md`.
This directory is the working area for making that direction concrete enough to
implement. Do not promote these chapters back into `docs/design/` until the PR
closeout merge.

The public lifecycle is:

1. authored orchestrate program
2. CMake target
3. run

The key rule is that CMake owns the offline build graph. Runtime C++ should run
the compiled orchestrate program directly, not load a module by string and not
fill a generic environment bag.

## Implementation-Ready Bar

A chapter is implementation-ready only when it names:

- the public API surface, if any;
- the internal records an implementation must build;
- the owner file or module for those records;
- the runtime path that consumes them;
- at least one verification check that can fail if the implementation drifts.

Concept explanations alone are not enough for this draft. If a term appears in
an example, the implementation chapters must say who materializes it, where it
lives, and how it is checked.

## Reading Order

- `00-overview.md`: entry point and reading order.
- `01-positioning.md`: why Megacu exists, the competitor map, and the target
  audience.
- `02-principles-and-naming.md`: naming model and thin-core rules.
- `03-program.md`: authored orchestrate-program model, named ops, resources,
  events, and domains.
- `04-cmake-build-and-runtime.md`: CMake-managed build graph and runtime stage
  boundaries.
- `05-language-responsibilities.md`: what belongs in authoring C++, CMake/native
  build rules, and runtime C++/CUDA.
- `06-compiled-orchestrate-program.md`: compiled orchestrate-program call
  surface.
- `07-dispatcher-scheduler-kernel.md`: build-graph mapping, scheduling, and
  kernel lowering.
- `08-examples.md`: concrete examples for authoring, CMake build, runtime, and
  framework integration.
- `09-first-validation-slice.md`: first narrow proof slice under the new
  boundary.
- `10-verification.md`: contracts, failure modes, and verification evidence.
- `11-distributed-launch-and-framework-integration.md`: CUDA+NVSHMEM
  multi-process launch, Torch Distributed integration, MPI integration, and
  symmetric allocation contracts.

Files `00` through `11` are the current active draft. They should be refined in
place until each contract, example, planned path, and verification requirement
is implementation-ready.

## Source Context

- MPK and Event Tensor source reading:
  `docs/notes/megakernel_cuda_layer_sources.md`
- Triton-distributed, MegaKittens, MSCCL++, and UniEP source reading:
  `docs/notes/distributed_backend_sources.md`
- FlashInfer framework-integration source reading:
  `docs/notes/framework_integration_sources.md`
- NVSHMEM and PyTorch distributed launch source reading:
  `docs/notes/distributed_launch_sources.md`
- PTO Runtime / simpler orchestration-surface reading:
  `docs/notes/orchestration_surface_sources.md`

## Current Design Direction

Megacu is a thin device-native composition layer, not a new compiler/runtime
stack.

- The public model is one authored orchestrate program plus explicit resources
  and events.
- Dispatcher, scheduler, kernel lowering, platform, and backend implementations
  are organized as reusable CMake build targets.
- Backend primitives remain callable inside native kernels.
- Runtime C++ calls the compiled orchestrate program directly.
- Megacu core records and lowering paths are C++/CUDA only. Reference framework
  adapters may live in this repository, but no framework tensor, process-group,
  or Python object may appear in `program_ir`, dispatcher plans, schedule plans,
  kernel plans, backend plans, or compiled target ABI records.
