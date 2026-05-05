# Current Thinness And Gaps

The current branch contains useful scaffolding, but the implementation is still
thin relative to the corrected architecture. This document lists the gaps that
must be closed before claiming a working runtime-linked Megacu slice.

## Current Checked-In Reality

- CMake still uses `megacu_add_components`, not the intended
  `megacu_add_configure_target`.
- `megacu_add_components` links `src/program/materialize.cc`,
  `src/lowering/static_stitch.cc`, and metadata section builders.
- Public authoring still exposes `program_builder` and `program_ir`.
- Runtime examples still validate linked metadata headers.
- Dispatcher and scheduler logic are implemented as static metadata builders,
  not runtime components.
- Target runtime is mostly example-local glue plus a link anchor.
- Golden/baseline/Megacu implementation roles are still blurry in native CUDA
  files.

These are facts about the current branch, not accepted architecture.

## Required Replacement Milestones

1. Add runtime headers for:
   - `runtime_context`;
   - `target_capability`;
   - `participant_attrs`;
   - `dispatch_request`;
   - `dispatch_state`;
   - scheduler operator wrappers.
2. Replace `megacu_add_components` with `megacu_add_configure_target` or make
   the existing function semantically equivalent.
3. Replace materializer-linked sources with runtime component sources.
4. Implement the annotated dispatcher:
   - maps participant annotations to tile/lane/peer work;
   - retargets `team_n_pes == 1` and multi-card teams.
5. Implement runtime schedulers:
   - ASAP scheduler consumes explicit dependency/event readiness;
   - static scheduler variants preserve explicit dependencies without queues.
6. Move common target runtime validation out of example headers into
   `src/target/`.
7. Implement backend runtime primitives and validation under
   `src/backends/nvshmem/`.
8. Implement CUDA launch feasibility validation under `src/platform/cuda/`.
9. Separate golden, baseline, and Megacu native paths in code.
10. Replace metadata tests with runtime component tests and CUDA/NVSHMEM
    correctness tests.

## Dispatcher-Specific Gaps

The dispatcher is the highest-risk component because it is responsible for
retargeting one `OrchTarget` across single-card and multi-card cases.

Missing evidence:

- same participant annotations map correctly for `team_n_pes == 1`;
- same participant annotations map correctly for `team_n_pes == 2`;
- communication participants become local/no-remote work in single-card mode;
- communication participants map to backend PEs in multi-card mode;
- unsupported annotation combinations return `megacu::status`.

## Scheduler-Specific Gaps

Missing evidence:

- ASAP scheduler permits AR tile execution after matching GEMM tile readiness;
- ASAP scheduler does not force a whole-GEMM-before-AR dependency unless
  capability requires it;
- static scheduler variants respect explicit dependency order.

## Backend And Platform Gaps

Missing evidence:

- NVSHMEM backend validates symmetric event and partial storage for multi-card
  runs;
- NVSHMEM backend permits local execution without remote peer requirements;
- device-side NVSHMEM primitive wrappers are used by baseline and Megacu paths;
- CUDA platform validates stream presence, device identity, and launch
  requirements.

## Non-Goals For This Documentation Slice

This documentation does not claim:

- the current code already satisfies the runtime-linked architecture;
- performance parity with MPK, Triton-Distributed, or handwritten kernels;
- zero-overhead execution;
- complete multi-backend retargetability.

Those claims require implementation and measurement evidence.
