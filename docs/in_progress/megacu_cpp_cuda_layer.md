# Feature Task: Megacu Device-Native Layer Design

- Branch: `main`
- PR:
- Owner: Codex
- Status: in progress

## Goal

Design the initial Megacu device-native layer as a thin native orchestration
model with CMake-managed strategy selection and a minimal runtime C++ API.

The active public lifecycle is:

`authored orchestrate program -> CMake target -> run`

The key implementation constraint is that CMake/build must not appear in the
runtime C++ API.

## Input

- User direction: build the repo harness before starting true Megacu design.
- User direction: multi-GPU support must be included from the beginning.
- User direction: use NVSHMEM first, but do not rebuild transport runtimes.
- User direction: study Triton-distributed, HazyResearch Megakernels,
  ThunderKittens, MSCCL++, UniEP, FlashInfer, and PTO Runtime / simpler where
  useful.
- User direction: Megacu should expose zero-overhead abstractions that land on
  existing CUDA/NVSHMEM/MSCCL++ mechanisms rather than replacing them.
- User direction: the design must not be tightly bound to CUDA even though CUDA
  is the first platform binding and performance baseline.
- User direction: structure the design docs rather than keeping a monolithic
  draft.
- User direction: shared concepts must be justified carefully and kept small.
- User direction: framework integration is required, but framework types are not
  part of the core Megacu layer.
- User direction: keep the layer tiny but mighty and use iterative refinement.
- User direction: keep the public model thin; use configuration or linked
  implementation to choose scheduler/kernel/backend behavior.
- User direction: tasks should be collected through a concise authoring surface,
  not a trait-heavy public API.
- User direction: the design must support fine-grained overlap without forcing a
  public fragment-op taxonomy.
- User direction: use `orchestrate` for runtime instantiation.
- User direction: Megacu itself should be C++ only, with no Python dependency
  in the core design.
- User direction: reusable engines should be compiled once, then a concrete
  orchestrate program should reuse those compiled artifacts in the build graph.
- User direction: flatten the design files and keep them ordered for easy
  browsing.
- User direction: avoid string-based module loading and generic runtime envs in
  the primary API; let CMake manage build and expose direct compiled
  orchestration instead.

## Output

- Stable entry point:
  `docs/in_progress/design/megacu_cpp_cuda_layer.md`
- Ordered design chapters under:
  `docs/in_progress/design/megacu_device_native_layer/`
- Redesign note:
  `docs/in_progress/design/megacu_device_native_layer/90-redesign-2026-04-24-cmake-build-and-direct-runtime.md`
- Updated source notes:
  `docs/notes/megakernel_cuda_layer_sources.md`
  `docs/notes/distributed_backend_sources.md`
  `docs/notes/framework_integration_sources.md`
  `docs/notes/orchestration_surface_sources.md`

## Scope Checklist

- [x] Define input, output, and verification criteria
- [x] Ask design-scope questions
- [x] Present two or three viable approaches with tradeoffs
- [x] Select one approach with user approval
- [x] Fill in the design draft
- [x] Map design examples to tests, benchmarks, profiler evidence, or
  generated-code inspection
- [x] Verify docs locally
- [ ] Finish redesign review against current CMake/build and runtime-surface
  concerns

## Verification

- `git status --short --branch`
- focused reread of the accepted design draft
- terminology and stale-path scan for superseded lifecycle names and moved files

## Tests

No code tests yet. The design must define future tests and evidence before
implementation starts.

## Docs

- `docs/in_progress/design/megacu_cpp_cuda_layer.md`
- `docs/in_progress/design/megacu_device_native_layer/`
- `docs/notes/megakernel_cuda_layer_sources.md`
- `docs/notes/distributed_backend_sources.md`
- `docs/notes/framework_integration_sources.md`
- `docs/notes/orchestration_surface_sources.md`

## Accepted Constraints

- Multi-GPU concepts are part of the initial design.
- The core API should stay thin and concise.
- Runtime C++ begins at the compiled orchestrate program.
- CMake/build belongs to offline native build flow, not the runtime C++ API.
- Scheduler choice, dispatcher choice, kernel lowering, platform/backend
  selection, and most id mapping are CMake/build-graph concerns.
- Related works are pressure tests, not mechanisms to copy one by one.
- Users should not normally author raw execution descriptors or trait-heavy task
  declarations.
- The primary API should not rely on string-path loading or generic runtime
  environments.

## Closeout

After design approval, commit the task and design draft. Implementation must
start from a written plan, not directly from this task file.
