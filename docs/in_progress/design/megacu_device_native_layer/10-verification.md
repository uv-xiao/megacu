# Verification

## Core Contracts

- The public authoring model is an orchestrate program, not a runtime-selected
  config object.
- The public runtime model is compiled orchestrate program -> `run(...)`.
- CMake/build does not appear in the runtime C++ API.
- Dispatcher, scheduler, kernel lowering, platform, and backend are selected by
  the build graph.
- Named kernels may use backend primitives inside the kernel body.
- Fine-grained overlap must remain possible without introducing a public
  fragment taxonomy.

## Failure Modes

- Runtime strategy selection: runtime C++ still chooses dispatcher, scheduler,
  kernel lowering, platform, or backend. This violates the build-graph strategy
  rule.
- Build leakage into runtime C++: runtime API exposes build/materialize work.
  This violates the language split.
- No artifact reuse: every concrete program recompiles dispatcher/scheduler/
  lowering engines instead of reusing compiled artifacts. This violates the
  CMake build-graph model.
- Plugin-shaped primary API: normal user code requires string path loading or a
  generic `runtime_env`. This violates the direct-compiled-runtime rule.
- Opaque-kernel trap: named ops cannot express fine-grained overlap because
  backend primitives are excluded from kernel bodies. This violates the overlap
  goal.
- Descriptor leakage: users are forced to author packed descriptors directly.
  This violates the program model.
- Mapping/scheduling conflation: dispatcher and scheduler are not clearly
  separated. This weakens implementation clarity.

## Verification Requirements

- CMake/build tests that produce reusable component targets
- CMake/build tests that produce one orchestrate target from those components
- native build checks that the artifact is produced by ordinary CUDA/C++
  compilation rather than runtime code generation
- direct-call smoke tests for the compiled orchestrate program
- generated-code inspection for the first CUDA/NVSHMEM target
- repeated-run tests showing the internal `run(...)` path stays cheap
- integration tests showing CMake/build drive target creation while runtime C++
  only calls the compiled orchestration
- two-rank runtime smoke tests for the first target where hardware is available

## Out Of Scope

- Runtime compilation through the C++ API
- String-path plugin loading as the primary API
- Public fragment-op taxonomies
- Public multiple scheduler APIs in the thin core
- Reimplementing NVSHMEM, MSCCL++, or other transport runtimes
