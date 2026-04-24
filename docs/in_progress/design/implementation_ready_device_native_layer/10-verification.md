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
- build evidence that `PROGRAM gemm_allreduce_program` metadata is produced by
  the native C++ build path, not by Python or runtime parsing
- design or compile evidence that each public concept in
  `03-program.md` is required by either scheduling, lowering, backend
  resolution, or runtime parameterization
- tests or inspection proving compiled function parameters fill pre-lowered
  slots and do not select dispatcher, scheduler, lowering, platform, or backend
  strategy
- native build checks that the artifact is produced by ordinary CUDA/C++
  compilation rather than runtime code generation
- checks proving kernel lowering and target lowering select/link existing
  implementations and materialize metadata rather than emit new C++/CUDA source
- direct-call smoke tests for the compiled orchestrate program
- linked-artifact or metadata inspection for the first CUDA/NVSHMEM target,
  including domain, participant, event, dispatch, schedule, and backend slots
- repeated-run tests showing the internal `run(...)` path stays cheap
- integration tests showing CMake/build drive target creation while runtime C++
  only calls the compiled orchestration
- two-rank GEMM+AllReduce correctness tests for the first target where hardware
  is available

## Example-To-Evidence Mapping

- Program API example in `03-program.md` and
  `09-first-validation-slice.md`:
  compile-only test under `tests/build/` plus direct-call smoke test.
- CMake examples in `04-cmake-build-and-runtime.md`:
  configure/build test proving component target reuse and orchestrate target
  linkage.
- Dispatcher, scheduler, and lowering contracts in
  `07-dispatcher-scheduler-kernel.md`:
  linked-artifact and metadata inspection showing op, resource, event,
  dispatch, participant, schedule, kernel, and backend payload ownership.
- Kernel context and backend primitive examples in `03-program.md` and
  `08-examples.md`:
  compile-only checks showing kernels use typed `kernel_context` APIs instead
  of string event lookup or hand-authored backend signal addresses.
- First slice runtime behavior in `09-first-validation-slice.md`:
  two-rank CUDA/NVSHMEM GEMM+AllReduce correctness test where hardware exists;
  explicit skip reason where local NVSHMEM multi-GPU execution is unavailable.
- Larger MPK-style example in `08-examples.md`:
  design/compile evidence that a serving-layer program can link CUDA-provided
  RMSNorm, linear, paged-attention, split-reduce, and residual-output operator
  bodies without changing the public program model.
- No runtime strategy selection:
  code inspection or test hook proving `run` does not call build, CMake,
  dispatcher selection, scheduler selection, backend selection, or string-based
  module loading.

## Ready-To-Promote Criteria

This active design is ready to merge back into `docs/design/` only when:

- every public surface has a planned owner and file path in the owning chapter;
- every example has a corresponding test, inspection, or skip rule;
- no stable doc points at unfinished draft content as implemented behavior;
- the first implementation slice can be built from the component contracts
  without introducing new public concepts.

## Out Of Scope

- Runtime compilation through the C++ API
- String-path plugin loading as the primary API
- Public fragment-op taxonomies
- Public multiple scheduler APIs in the thin core
- Reimplementing NVSHMEM, MSCCL++, or other transport runtimes
