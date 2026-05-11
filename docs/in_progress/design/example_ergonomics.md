# Design: Example Ergonomics

## Goal

Make Megacu examples show the intended authoring model:

- write small operator kernels;
- write a recipe that submits operator tasks and sync tasks;
- select a runtime strategy;
- let Megacu own the repeated arena, launch, operator-table, and wrapper
  scaffolding.

The examples should not teach users that every target must hand-write a large
`_arena.cuh`, `megacu.cu`, and `orchestrate.cc` stack.

## Context

The current examples validate the architecture, but they expose too much of the
runtime plumbing in every example directory. This was acceptable while proving
the runtime-linked model, but it is now the main usability problem:

- repeated arena structs make capacity and buffer ownership look like target
  logic;
- repeated host-orch/seeded-orch wrappers make runtime selection look
  example-specific;
- repeated CUDA launch glue makes operator kernels look harder to write than
  they are;
- tutorials must explain too many support files before the operator-task model
  is visible.

## Alternatives

1. Keep each example explicit and document the pattern better.
   - Simple for implementation.
   - Bad for user experience: examples stay large and intimidating.

2. Add a common example helper layer under `examples/cuda_nvshmem/common/`.
   - Shrinks examples quickly.
   - Risks creating an example-only mini framework that does not improve
     Megacu itself.

3. Add thin Megacu-owned runtime authoring helpers.
   - Moves repeated mechanics into the library surface.
   - Keeps examples focused on recipes and operators.
   - Requires careful contracts so helpers stay thin and do not hide target
     behavior.

## Selected Design

Use option 3.

Add a thin Megacu-owned authoring layer that provides reusable building blocks
for runtime-linked CUDA examples without inventing a high-level graph system.
The target still owns operator kernels and task recipes. Megacu owns repeated
mechanics that are not target semantics:

- fixed-capacity arena storage wrappers;
- host-orch and seeded-orch wrapper generation;
- CUDA operator-table and launch adapter glue;
- common validation/launch status propagation helpers.

The API should stay direct and CUDA-like. Arguments remain raw pointers and
plain target structs. The helper should not infer dependencies from tensors,
perform fallback checking, or split tasks into sub-tasks.

## Examples

- Example: GEMM-RS
- Feature shown: user writes `build_runtime_recipe(...)` and tile operator
  kernels; Megacu helper materializes host-orch and seeded-orch entrypoints.
- Verification mapping: existing `cuda_gemm_reduce_scatter_correctness` and
  two-rank GEMM-RS tests pass unchanged, while static checks confirm generic
  arena/wrapper boilerplate no longer lives in the example.

- Example: AG-GEMM
- Feature shown: arbitrary-K dependency groups and EventTensor waits remain
  explicit recipe attributes, but repeated launch/orchestrate files are
  removed or reduced to declarations.
- Verification mapping: existing uneven-K distributed tests pass unchanged.

- Example: tiny decode
- Feature shown: non-collective multi-rank pipeline can use the same helper
  shape as GEMM examples.
- Verification mapping: local and MPI tiny decode tests pass unchanged.

## Contracts

- Operator kernels use a common thin API and do not call backend-specific macro
  protocols directly.
- Recipes submit tasks, dependencies, and EventTensor attrs explicitly.
- Runtime wrappers are Megacu-owned, not copy-pasted per example.
- Examples may still own golden/baseline code and workload-specific validation.
- No heavy validation or fallback checking is added.

## Failure Modes

- If the helper hides too much, examples stop showing the architecture.
- If the helper stays example-only, it becomes duplicated framework code.
- If capacity policy is implicit, users lose responsibility for choosing
  enough arena space.

## Verification

Required:

- local build and CTest;
- Docker CUDA+NVSHMEM two-card gate;
- static checks that active examples do not duplicate generic arena and wrapper
  scaffolding;
- tutorial review showing a beginner sees operator-task recipe first.

## Out Of Scope

- A high-level tensor graph builder.
- Automatic dependency inference from tensor access.
- Runtime fallback checking for missed dependencies or invalid user buffers.
- Benchmark claims.

## Closeout

Promote the accepted API and example-authoring rules into `docs/design/`, then
remove this in-progress design draft before merging.
