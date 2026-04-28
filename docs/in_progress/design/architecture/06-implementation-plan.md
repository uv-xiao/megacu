# PR #4 Architecture Repair Plan

This plan is intentionally narrower than the previous concrete implementation
plan. PR #4 should fix the architecture design and keep only one tiny proof
example in scope. General concrete implementation work now lives in
`docs/todo/concrete_impl/`.

## Split The Work

PR #4 owns:

- the runtime-linked architecture correction;
- the direct ABI and typed runtime-view story;
- the component ownership boundaries;
- one tiny but mighty example that may be problem-specific;
- design evidence that the path no longer depends on compiler-like
  materialization.

Future TODO work owns:

- reusable `ConfigureTarget` and `OrchTarget` CMake helpers;
- general annotation-driven dispatcher implementation;
- reusable scheduler runtime families;
- complete CUDA+NVSHMEM GEMM+AllReduce phased and overlap examples;
- MPI and torch-distributed adapters;
- broad include/src cleanup and verification.

## Repair Architecture Docs First

Review these files as one architecture packet:

- `00-overview.md`
- `01-programming-surface.md`
- `02-build-link-config.md`
- `03-runtime-components.md`
- `04-distributed-runtime.md`
- `05-gemm-allreduce-example.md`
- `07-verification.md`

The review should check:

- no accepted path requires `program_ir`, `materialize_program`, static
  metadata sections, or generated source;
- the tiny example is clearly allowed to be problem-specific;
- problem-specific proof shortcuts are kept out of public API and shared
  component contracts;
- future general implementation requirements point to
  `docs/todo/concrete_impl/`;
- distributed and framework concerns are architecture requirements, not hidden
  PR #4 implementation promises.

## Tiny Proof Example Bar

If PR #4 includes code, keep the example small enough to inspect end to end:

- one direct orchestrate ABI;
- one narrow problem/workspace type;
- one linked native operator symbol;
- one runtime mapping step from problem/team values;
- one runtime progress step that calls the operator;
- status propagation for invalid runtime views.

The proof must not introduce a new generic runtime graph, program builder,
materializer, generated metadata format, or runtime component registry.

## Handoff To Future Concrete Implementation

The future implementation work should start from `docs/todo/concrete_impl/`.
Before that work begins, those docs must be reviewed against the active
architecture and strengthened where needed so shared runtime components are
general, reusable, and tested with at least one nontrivial reuse check.
