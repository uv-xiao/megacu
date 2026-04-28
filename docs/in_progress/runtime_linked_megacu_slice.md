# Feature Task: Runtime-Linked Megacu Slice

- Branch: `implementation/runtime-linked-megacu-slice`
- PR: #4
- Owner: Codex
- Status: Active, split to architecture redesign

## PR #4 Scope

PR #4 now focuses on two things:

- fix the runtime-linked architecture design after the compiler-like
  materialization path was rejected;
- include one very tiny but mighty proof example that may be problem-specific
  and does not need to prove reusable generality.

The broad concrete implementation work moved to
`docs/todo/concrete_impl/`. That future work must satisfy strong generality
requirements before it returns to active implementation scope.

## Review Reset

PR #4 continues the PR #3 review reset. It must no longer pursue the
compiler-like implementation that records program IR, materializes owned facts,
builds dispatch/schedule/kernel/backend sections, and validates linked target
metadata. That model is too heavy and does not match the intended Megacu layer.

The active design source is `docs/in_progress/design/architecture/`. The
previous stable copy was moved back from `docs/design/` because it is not
accepted implemented behavior.

## Corrected Design Contract

Megacu is a runtime-linked device-native layer:

- CMake links selected dispatcher, scheduler, platform, backend, target runtime,
  and native operator implementations.
- Public C++ exposes direct orchestrate ABI functions and typed runtime views.
- Runtime components are called inside the orchestrate target rather than
  generated from a host materializer.
- Platform and backend adapters validate native CUDA/NVSHMEM handles and expose
  runtime/device-side primitives.
- Native CUDA/NVSHMEM operators execute the actual work.
- No normal implementation path uses `program_ir`, `owned_program_ir`,
  `materialize_program`, `target_metadata`, static `dispatch_section`, static
  `schedule_section`, or host materializer executables.

The architecture may describe where reusable `ConfigureTarget` dispatcher and
scheduler components belong. PR #4 does not need to implement that reusable
generality. Its proof example may use problem-specific mapping and scheduling
helpers as long as they are named as example-local and do not become public
Megacu API.

## Tiny Proof Example

The PR #4 example should be as small as possible while still exercising the
architecture:

- one direct orchestrate ABI;
- typed runtime views for launch, team, event storage, and problem/workspace;
- one linked native operator path;
- one explicit runtime mapping step from problem/team values to work;
- one explicit runtime progress step that calls the linked operator;
- clear status propagation before launch on invalid runtime views.

The example may be a narrow GEMM+AllReduce case, a single tile, a fixed dtype,
or another problem-specific shape if it proves the architecture more directly.
It should not try to become the general dispatcher, scheduler, or full
CUDA+NVSHMEM example suite.

## Required File Organization For PR #4

Active design ownership:

- `docs/in_progress/design/architecture/`: corrected runtime-linked design.
- `docs/in_progress/runtime_linked_megacu_slice.md`: PR #4 task scope.
- `docs/todo/concrete_impl/`: future general implementation requirements.

If the tiny proof includes code, keep problem-specific shortcuts in example or
test-owned files. Shared `include/megacu/` and `src/` files should only contain
contracts that the architecture is ready to defend beyond the tiny proof.

## Out Of Scope For PR #4

The following are future `docs/todo/concrete_impl/` work unless explicitly
reintroduced after design review:

- full reusable annotation-driven dispatcher implementation;
- full phased and overlap scheduler families;
- complete CUDA+NVSHMEM GEMM+AllReduce baseline matrix;
- MPI and torch-distributed adapters;
- broad public API cleanup across `include/` and `src/`;
- performance claims or zero-overhead claims;
- multi-backend generality.

## Current Gap List

- Tighten the architecture docs so they separate final architecture direction
  from PR #4 proof-example scope.
- Keep the tiny example problem-specific where that reduces design noise.
- Move concrete implementation details and generality gates to
  `docs/todo/concrete_impl/`.
- Remove stale active-doc references to
  `docs/in_progress/design/concrete_impl/`.
- Preserve the rejection of compiler-like materialization as a non-negotiable
  architecture correction.

## Verification For This Documentation Split

Minimum evidence for this split:

```sh
rg -n "docs/in_progress/design/concrete_impl|PR #4 must replace|concrete implementation docs for PR #4" \
  docs/in_progress docs/todo
git diff --check
```

Documentation-only changes do not require automated tests. The handoff should
record focused rereads and path checks instead of claiming implementation
behavior.
