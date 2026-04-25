# Implementation Plan

This plan replaces the compiler-like implementation with runtime-linked
components.

## Remove Compiler-Like Pieces

Delete or replace:

- `include/megacu/detail/program_ir.h`
- `include/megacu/detail/materialize.h`
- `include/megacu/detail/target_metadata.h` as the main target contract
- `src/program/materialize.cc`
- section-building functions in `src/dispatcher/`, `src/scheduler/`,
  `src/lowering/`, and `src/backends/nvshmem/`
- tests that assert materialized IR or static metadata sections.

If any tiny linked config remains, it must be capability data, not dispatch or
schedule data.

## Add Runtime Contracts

Add or repurpose headers around:

```text
include/megacu/runtime_config.h
include/megacu/runtime_context.h
include/megacu/detail/component_runtime.h
```

Expected concepts:

- `runtime_context`: launch, team, event storage, optional target capability.
- `target_capability`: linked static envelope for supported backend/platform,
  dtype/layout/team sizes, and scheduler mode.
- `dispatch_state` and `tile_work`: compact runtime work cursors.
- scheduler entrypoints for phased and overlap GEMM+AllReduce.

Do this before changing the example runtime path. The old materializer tests
should be replaced by tests for these contracts in the same patch series, not
left as stale evidence.

## Runtime Dispatcher

Replace the current ad-hoc dispatcher with a GEMM+AllReduce runtime dispatcher:

- input: `gemm_ar_problem`, `team_view`, target capability;
- output: compact `dispatch_state`;
- helper: iterate tile work without heap-heavy tables;
- behavior: map logical rank and peer rank from runtime team values.

## Runtime Scheduler

Replace static schedule-section construction with runtime scheduler calls:

- phased scheduler executes compute/communication according to tile readiness;
- overlap scheduler enforces co-resident/persistent progress guard;
- both schedulers call linked native operator symbols.

Scheduler tests should use small fake operator structs that record which path
was called. Numeric CUDA tests then prove the real operator path.

## Backend And Platform

NVSHMEM backend:

- validate team size and PE identity;
- validate symmetric event and partial storage;
- expose device-side primitive wrappers or linked helpers.

CUDA platform:

- validate stream/device;
- validate persistent/cooperative launch requirements for overlap;
- convert CUDA errors to `megacu::status`.

## Distributed Coverage

Add implementation paths or stubs with explicit blockers for:

- `nvshmrun` two-card execution;
- MPI session adapter;
- torch-distributed session adapter.

The first PR may keep MPI and torch adapters minimal, but the design and test
plan must show how they feed the same `launch_view` and `team_view` into the
direct orchestrate ABI.

Concrete PR expectation:

- keep `nvshmrun`/Docker two-card as executable evidence;
- add MPI adapter headers or an explicit skipped integration test with the
  missing dependency named;
- add torch-distributed adapter design stubs or an explicit skipped integration
  test with missing dependency named;
- do not claim those integrations work until their smoke tests run.

## Update Existing Docs

The following files must be updated during implementation:

- `docs/in_progress/design/concrete_impl/`: replace materializer call path with
  runtime-linked component call path.
- `docs/in_progress/public_builder_surface.md`: keep it aligned with this
  design.
- Example READMEs: remove "materialized schedule metadata" claims and document
  runtime dispatcher/scheduler behavior.
- PR body: describe current implementation honestly after each batch.

## Documentation Updates

After code replacement:

- update `docs/in_progress/design/concrete_impl/` so it documents runtime
  components, not materialization;
- update `docs/in_progress/public_builder_surface.md` to remove materializer
  requirements and add distributed runtime requirements;
- keep `docs/design/` untouched until this redesigned work is accepted and
  ready to merge.
