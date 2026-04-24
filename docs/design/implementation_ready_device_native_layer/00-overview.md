# Implementation-Ready Megacu Device-Native Layer Design

This directory holds the accepted implementation-ready Megacu device-native
design in one flat, ordered set of files. It is the implementation contract set
for the picked direction in `docs/design/megacu_cpp_cuda_layer.md`.

This directory intentionally avoids broad positioning, audience, and competitor
analysis. It contains the API surfaces, internal records, owner paths,
examples, failure checks, and verification evidence needed to start
implementation.

## Implementation-Ready Bar

A chapter is implementation-ready only when it names:

- the public API surface, if any;
- the internal records an implementation must build;
- the owner file or module for those records;
- the runtime path that consumes them;
- at least one verification check that can fail if the implementation drifts.

Concept explanations alone are not enough for this draft. Keep term definitions
only when they are necessary to understand an implementation contract.

## Thinness Rule

This design must keep Megacu thinner than MPK, Triton-Distributed, and
MegaKittens-style systems at the user-facing layer. The first implementation
public surface is limited to:

- `program_builder` and small helpers for domains, participants, resources,
  events, and submissions;
- typed runtime views needed by the compiled target;
- `megacu_add_components(...)` and `megacu_add_orchestrate_target(...)`;
- the compiled orchestrate function that applications call directly.

Dispatcher, scheduler, kernel-lowering, platform, backend, and metadata
records are private implementation details for one target. They should be
stored as component-owned sections inside one linked target metadata blob, not
as public APIs, runtime lifecycle objects, or user-visible layers.

Any new public view, plan, record, lifecycle term, or backend taxonomy must pass
one of these tests:

- it is required by the first GEMM+AllReduce implementation slice;
- it is required by the second MPK-style validation target without duplicating
  an existing concept;
- two independent targets need the same public surface and cannot keep it
  target-specific.

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

These chapters are the starting contract for implementation. Implementation
PRs may refine them when concrete code or verification exposes a mismatch.
