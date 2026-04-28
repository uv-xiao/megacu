# TODO: General Runtime-Linked Concrete Implementation

These documents describe future concrete implementation work for the
runtime-linked Megacu architecture. They were moved from
`docs/in_progress/design/concrete_impl/` to `docs/todo/concrete_impl/` on
2026-04-28 when PR #4 was split.

They are not PR #4 scope. PR #4 should focus on architecture repair and one
tiny but mighty proof example that may be problem-specific. This TODO
workstream is the later implementation bar for making the runtime-linked design
general.

They are not a stable design and they are not a description of accepted
implemented behavior. They are implementation-facing requirements for replacing
the current compiler-like path with general runtime-linked components.

## Generality Requirements

Future concrete implementation work must not treat the PR #4 tiny example as
the reusable architecture. Before this work returns to active implementation
scope, it must satisfy these requirements:

- The shared runtime components must be reusable `ConfigureTarget` components,
  not GEMM+AllReduce-only code moved into shared directories.
- Problem-specific code belongs in `OrchTarget` example files: direct ABI,
  problem/workspace types, participant annotations, native operator symbols,
  and example-local validation.
- Dispatcher APIs must be driven by typed participant annotations plus runtime
  problem/team views. A dispatcher may have platform/backend-specific
  algorithms, but it must not depend on example names, debug strings, or
  hard-coded GEMM+AllReduce roles.
- Scheduler APIs must consume dispatch state and linked operator capabilities,
  not recompute rank/peer placement or depend on one example's tensor layout.
- Platform and backend validation must be independent of the example workload.
  CUDA and NVSHMEM code may be first, but their public views must not encode
  GEMM+AllReduce assumptions.
- CMake target helpers must reject unknown component names and unsupported
  combinations without silently falling back to example-specific defaults.
- Verification must include at least one nontrivial reuse check: either a
  second small `OrchTarget`, a fake-workload component test, or an explicit
  contract test proving the same dispatcher/scheduler API accepts a different
  participant set.
- Any remaining problem-specific shortcut must be named as temporary and kept
  out of public Megacu APIs.

## Current Correction

The checked-in implementation still contains transitional files such as:

- `include/megacu/program.h`
- `include/megacu/detail/program_ir.h`
- `include/megacu/detail/materialize.h`
- `include/megacu/detail/target_metadata.h`
- `src/program/materialize.cc`
- metadata-section builders under `src/dispatcher/`, `src/scheduler/`,
  `src/lowering/`, and `src/backends/nvshmem/`

Those files are useful evidence of what must be replaced, but they are not the
intended implementation contract. The intended implementation is:

```text
direct OrchTarget ABI
  -> typed runtime views and target capability
  -> virtual participant annotations from the OrchTarget
  -> ConfigureTarget runtime dispatcher maps participants/problem/team
  -> scheduler consumes dispatch state and chooses progress actions
  -> platform/backend validate native resources
  -> linked native CUDA/NVSHMEM operators run
```

## Scope

This directory owns concrete implementation documentation for:

- public headers under `include/megacu/`;
- private runtime component headers under `include/megacu/detail/`;
- source ownership under `src/dispatcher/`, `src/scheduler/`,
  `src/platform/`, `src/backends/`, and `src/target/`;
- CMake linking in `cmake/MegacuTargets.cmake`;
- CUDA+NVSHMEM GEMM+AllReduce example call paths.

It must stay aligned with:

- `docs/in_progress/design/architecture/`;
- `docs/in_progress/runtime_linked_megacu_slice.md`;
- example organization rules under `.agents/rules/example-organization.md`.

## Reading Order

1. `01-file-map.md`: intended file/module ownership and transitional files to
   remove or repurpose.
2. `02-build-and-linking.md`: concrete CMake target shape for
   `ConfigureTarget` and `OrchTarget`.
3. `03-runtime-call-path.md`: call-by-call runtime path for phased and overlap
   GEMM+AllReduce.
4. `04-component-contracts.md`: concrete implementation contracts per
   component.
5. `05-current-thinness-and-gaps.md`: current gaps and replacement milestones.
