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

This README is the current TODO contract. The child files were moved from the
older concrete-implementation draft and may still contain stale names such as
fixed direct ABIs, `participant_attrs`, or platform/backend runtime views. Any
future implementation pass must update those child files to match the
architecture in `docs/design/runtime_linked_device_native_layer/` before using
them as a plan.

## Generality Requirements

Future concrete implementation work must not treat the PR #4 tiny example as
the reusable architecture. Before this work returns to active implementation
scope, it must satisfy these requirements:

- The shared runtime components must be reusable `ConfigureTarget` components,
  not GEMM+AllReduce-only code moved into shared directories.
- Problem-specific code belongs in `OrchTarget` example files: argument schema,
  workload orchestration code, operator schemas, component attributes, native
  operator symbols, and example-local validation.
- Workload ABI arguments must be configurable by target schema. Future concrete
  implementation must not standardize fixed arguments such as workspace,
  events, CUDA launch view, NVSHMEM team view, and problem.
- Platform/backend resources must live in component-owned driver resources, not
  Megacu core runtime-view fields or target arguments.
- Dispatcher APIs must be driven by submitted tasks and typed
  component-provided attributes. A dispatcher may have platform/backend-specific
  algorithms, but it must not depend on example names, debug strings, or
  hard-coded GEMM+AllReduce roles.
- Scheduler APIs must consume dispatch state and linked operator capabilities,
  plus task dependencies represented as explicit scheduler attributes. They
  must not infer dependencies from input/output/inout declarations, tensor
  lookup, pointer aliasing, or one example's tensor layout, and must not
  recompute rank/peer placement. Missing dependency attributes are target-author
  errors; Megacu should not add fallback dependency inference.
- Platform and backend validation must be independent of the example workload.
  CUDA and NVSHMEM code may be first, but their public views must not encode
  GEMM+AllReduce assumptions.
- Operator boundaries must support configurable argument schemas rather than
  one global fixed native signature.
- CMake target helpers must reject unknown component names and unsupported
  combinations without silently falling back to example-specific defaults.
- Verification must include at least one nontrivial reuse check: either a
  second small `OrchTarget`, a fake-workload component test, or an explicit
  contract test proving the same dispatcher/scheduler API accepts a different
  participant set.
- Any remaining problem-specific shortcut must be named as temporary and kept
  out of public Megacu APIs.

## Current Correction

PR #4 removed the compiler-like program/materializer path from the active
implementation. Future concrete implementation should build on the
runtime-linked contract:

```text
OrchTarget entry
  -> driver plus target argument schema
  -> submitted tasks with raw arguments and explicit dependency attributes
  -> component-provided attributes from the OrchTarget
  -> ConfigureTarget runtime dispatcher maps tasks/resources
  -> scheduler consumes dispatch state and explicit dependency attributes
  -> platform/backend expose thin driver facts and device primitives
  -> one Megacu mega-kernel uses linked scheduler, dispatcher, backend, and
     operator-table APIs
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

- `docs/design/runtime_linked_device_native_layer/`;
- example organization rules under `.agents/rules/example-organization.md`.

## Reading Order

1. `01-file-map.md`: intended file/module ownership and transitional files to
   remove or repurpose.
2. `02-build-and-linking.md`: concrete CMake target shape for
   `ConfigureTarget` and `OrchTarget`.
3. `03-runtime-call-path.md`: call-by-call runtime path for phased
   GEMM+AllReduce.
4. `04-component-contracts.md`: concrete implementation contracts per
   component.
5. `05-current-thinness-and-gaps.md`: current gaps and replacement milestones.
