# PR #4 Architecture Repair Plan

This plan is intentionally narrower than the previous concrete implementation
plan. PR #4 should fix the architecture design and keep only one tiny proof
example in scope. General concrete implementation work now lives in
`docs/todo/concrete_impl/`.

## Split The Work

PR #4 owns:

- the runtime-linked architecture correction;
- the direct orchestrate argument signature and driver abstraction story;
- the component ownership boundaries;
- one tiny phased example that may be problem-specific;
- 1-host/1-device and 1-host/2-device coverage for that example;
- design evidence that the path no longer depends on compiler-like
  materialization.

Future TODO work owns:

- reusable `ConfigureTarget` and `OrchTarget` CMake helpers;
- full component-provided attribute vocabulary and general dispatcher
  implementation;
- reusable scheduler runtime families;
- complete reusable CUDA+NVSHMEM GEMM+AllReduce examples;
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
- workload ABI arguments are configurable and do not expose CUDA/NVSHMEM
  handles as fixed target arguments;
- programmers do not write dispatcher/scheduler boilerplate;
- component attributes are extensible and component-provided;
- operator calls pass raw arguments to linked native signatures without
  `input`/`output`/`inout` wrappers;
- `submit` uses the linked operator signature arity to separate raw operator
  arguments from an optional trailing attribute set;
- task dependencies are explicit dependency attributes and are not inferred from
  submitted tensor accesses, and Megacu does not fallback-check missing edges;
- the tiny example is phased-only and clearly allowed to be problem-specific;
- problem-specific proof shortcuts are kept out of public API and shared
  component contracts;
- 1-host/1-device and 1-host/2-device are PR #4 gates;
- future general implementation requirements point to
  `docs/todo/concrete_impl/`;
- MPI and torch-distributed are future work, not hidden PR #4 gates.

## Tiny Proof Example Bar

If PR #4 includes code, keep the example small enough to inspect end to end:

- one orchestrate entry taking a driver plus direct CUDA-kernel-like target
  arguments;
- one narrow direct target argument signature;
- one or two linked native operator symbols for phased GEMM+AllReduce;
- one runtime mapping step from submitted tasks and driver resources;
- one ASAP scheduler step that consumes the explicit GEMM-to-sync-to-AR
  dependency path
  attribute;
- one Megacu megakernel host call whose common device-side running loop may
  dispatch and schedule multiple linked operator kernels internally;
- status propagation for failures reported by linked native operators;
- 1-host/1-device and 1-host/2-device validation.

The proof must not introduce a new generic runtime graph, program builder,
materializer, generated metadata format, or runtime component registry.

## Handoff To Future Concrete Implementation

The future implementation work should start from `docs/todo/concrete_impl/`.
Before that work begins, those docs must be reviewed against the active
architecture and strengthened where needed so shared runtime components are
general, reusable, and tested with at least one nontrivial reuse check.
