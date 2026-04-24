# Implementation Contracts

This chapter is the implementation bridge for the active design. It records
the concrete surfaces to build first and the evidence that will decide whether
the design is ready to move back into `docs/design/`.

## Selected Approach

Use the smallest direct-compiled path:

1. C++ users author kernels and one orchestrate function.
2. CMake builds reusable Megacu component targets.
3. CMake builds the orchestrate target by linking those reusable components.
4. Runtime C++ calls the compiled orchestrate function directly.

Rejected primary paths:

- runtime string loading of Megacu modules
- generic runtime environment bags
- runtime C++ build or compile APIs
- public task-trait field filling
- public scheduler or kernel-shape family APIs

Optional deployment ABIs can be reconsidered later, but they are not part of
the first implementation contract.

## Planned Repository Surfaces

The first implementation should keep ownership narrow:

- `include/megacu/orchestrator.h`: public authoring builders for domains,
  events, submissions, and `run`.
- `include/megacu/views.h`: typed resource views used by public examples.
- `include/megacu/backends/nvshmem.h`: explicit NVSHMEM handle/view adapters
  for the first backend.
- `cmake/MegacuTargets.cmake`: CMake functions for component and orchestrate
  targets.
- `src/dispatcher/`: reusable dispatcher implementations.
- `src/scheduler/`: reusable scheduler implementations.
- `src/lowering/`: reusable kernel-lowering implementations and target
  lowering driver.
- `src/platform/cuda/`: CUDA platform adapter code.
- `src/backends/nvshmem/`: NVSHMEM backend adapter code.
- `examples/cuda_nvshmem_event_copy/`: first executable proof.
- `tests/build/`: compile/link checks for the CMake target boundary.
- `tests/runtime/`: direct-call smoke tests, with multi-GPU/NVSHMEM tests
  skipped explicitly when hardware is unavailable.

These paths are design targets, not a requirement to create every directory in
the first code PR.

## Public C++ Contract

The public authoring contract is one small builder surface:

```cpp
void cuda_nvshmem_event_copy_orchestrate(
    megacu::workspace_view workspace,
    megacu::event_storage_view events,
    megacu::nvshmem_team_view team,
    std::int32_t tiles) {
  megacu::orchestrator orch{workspace, events, team};

  auto tile = orch.domain("tile", tiles);
  auto ready = orch.event("ready", tile, megacu::remote_event{});

  orch.submit(
      ops::write_then_signal,
      megacu::over(tile),
      megacu::args()
          .workspace(workspace)
          .signal(ready.release()));

  orch.submit(
      ops::wait_then_check,
      megacu::over(tile),
      megacu::args()
          .wait(ready.acquire())
          .workspace(workspace));

  orch.run();
}
```

Required properties:

- `orchestrator` construction takes explicit typed views and backend handles.
- `domain` names logical work only; it does not expose final worker ids.
- `event` creates a typed coordination object; release/acquire are dependencies
  attached to submissions.
- `submit` records named ops, domain membership, arguments, and dependencies.
- `run` launches the already-lowered target path for the current invocation.

Forbidden in the public contract:

- user-authored raw packed descriptors
- public `task_kind` or `task_desc` authoring structs
- public scheduler base classes
- public coordinate/rank id plumbing for normal examples
- generic string lookup for resources, backends, or modules

## CMake Contract

The CMake contract should expose two layers:

```cmake
megacu_add_components(
  NAME cuda_nvshmem_static
  DISPATCHER tile_dispatch
  SCHEDULER static_persistent
  KERNEL_LOWERING persistent_stitch
  PLATFORM cuda
  BACKEND nvshmem
)

megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_event_copy
  SOURCES event_copy_orchestrate.cc
  KERNELS event_copy_kernels.cu
  COMPONENTS cuda_nvshmem_static
)
```

Required properties:

- component targets are reusable build artifacts;
- orchestrate targets depend on component targets;
- strategy choice happens in CMake/native build metadata;
- runtime C++ cannot choose a different dispatcher, scheduler, lowering,
  platform, or backend for that target;
- the build can emit inspection metadata for generated/lowered code.

## Internal Lowering Contract

The orchestrate target lowering may create internal records, but those records
are not public authoring APIs.

Minimum internal records for the first slice:

- op table: named op symbol, implementation symbol, domain reference
- resource table: typed view slots used by lowered code
- event table: event storage, scope, release/acquire dependencies
- dispatch table: logical domain to execution placement
- schedule payload: static persistent execution order
- kernel payload: named kernel entrypoints and stitched execution metadata
- backend payload: NVSHMEM handles and signal/wait metadata needed by kernels

Each record must have one owner:

- public orchestrator builders collect semantic facts;
- dispatcher owns placement;
- scheduler owns execution order;
- lowering owns kernel stitching and launch payloads;
- platform/backend adapters own native handles and primitive calls.

No record should duplicate a concept already owned by another layer.

## First Slice Contract

The first implementation slice should be `cuda_nvshmem_event_copy`.

It must prove:

- two named CUDA kernels can be submitted through the public builder;
- one release/acquire event dependency crosses rank/GPU ownership;
- CMake builds reusable CUDA/NVSHMEM static components once;
- CMake builds one orchestrate target that reuses those components;
- runtime C++ calls the orchestrate function directly;
- the repeated execution path does not run build or strategy-selection logic.

Minimal runtime behavior:

1. rank A writes a payload;
2. rank A signals a remote event;
3. rank B waits on the event;
4. rank B verifies payload visibility.

## Example-To-Evidence Mapping

- Public C++ contract example:
  compile-only test under `tests/build/` plus direct-call smoke test.
- CMake contract example:
  CMake configure/build test proving component target reuse and orchestrate
  target linkage.
- Internal lowering contract:
  generated/lowered metadata inspection showing op, resource, event, dispatch,
  schedule, kernel, and backend payload ownership.
- First slice runtime behavior:
  two-rank CUDA/NVSHMEM test where hardware exists; explicit skip reason where
  local NVSHMEM multi-GPU execution is unavailable.
- No runtime strategy selection:
  code inspection or test hook proving `run` does not call build, CMake,
  dispatcher selection, scheduler selection, backend selection, or string-based
  module loading.

## Ready-To-Promote Criteria

This active design is ready to merge back into `docs/design/` only when:

- every public surface above has a planned owner and file path;
- every example has a corresponding test, inspection, or skip rule;
- no stable doc points at unfinished draft content as implemented behavior;
- the first implementation slice can be built from the contracts without
  introducing new public concepts.
