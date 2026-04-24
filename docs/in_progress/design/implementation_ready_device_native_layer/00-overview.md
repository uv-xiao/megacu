# Implementation-Ready Megacu Device-Native Layer Design

This directory holds the active implementation-ready Megacu device-native
design in one flat, ordered set of files. It is an implementation contract set,
not the stable architecture narrative.

The picked stable direction remains in `docs/design/megacu_cpp_cuda_layer.md`.
Do not duplicate positioning, audience, competitor analysis, or stable
principles here. This directory should contain only the API surfaces, internal
records, owner paths, examples, failure checks, and verification evidence needed
to implement that direction. Keep `docs/design/` untouched until the PR
closeout merge.

## Implementation-Ready Bar

A chapter is implementation-ready only when it names:

- the public API surface, if any;
- the internal records an implementation must build;
- the owner file or module for those records;
- the runtime path that consumes them;
- at least one verification check that can fail if the implementation drifts.

Concept explanations alone are not enough for this draft. Keep term definitions
only when they are necessary to understand an implementation contract.

## Reading Order

- `00-overview.md`: entry point and reading order.
- `01-program.md`: authored orchestrate-program model, named ops, resources,
  events, and domains.
- `02-cmake-build-and-runtime.md`: CMake-managed build graph and runtime stage
  boundaries.
- `03-language-responsibilities.md`: what belongs in authoring C++, CMake/native
  build rules, and runtime C++/CUDA.
- `04-compiled-orchestrate-program.md`: compiled orchestrate-program call
  surface.
- `05-dispatcher-scheduler-kernel.md`: build-graph mapping, scheduling, and
  kernel lowering.
- `06-examples.md`: concrete examples for authoring, CMake build, runtime, and
  framework integration.
- `07-first-validation-slice.md`: first narrow proof slice under the new
  boundary.
- `08-verification.md`: contracts, failure modes, and verification evidence.
- `09-distributed-launch-and-framework-integration.md`: CUDA+NVSHMEM
  multi-process launch, Torch Distributed integration, MPI integration, and
  symmetric allocation contracts.
- `10-implementation-architecture.md`: cross-cutting implementation guardrails
  for metadata, status, op symbols, dependencies, and error boundaries.

These chapters should be refined in place until each contract, example, planned
path, and verification requirement is implementation-ready.
