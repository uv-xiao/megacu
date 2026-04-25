# Concrete Implementation Docs

These documents translate the active architecture in
`docs/in_progress/design/architecture/` into concrete implementation ownership
for PR #4.

They are not a stable design and they are not a description of accepted
implemented behavior. They are implementation-facing notes for replacing the
current compiler-like path with runtime-linked components.

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
