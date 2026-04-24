# First Validation Slice

The first proof should validate the CMake-managed build graph and compiled
orchestrate-program surface.

## Goal

Prove all of the following with the smallest useful system:

- named kernels remain native CUDA/NVSHMEM code
- CMake reusable component targets produce dispatcher/scheduler/kernel-lowering
  engines
- CMake orchestrate-target compilation reuses those compiled artifacts
- runtime C++ calls the compiled orchestrate program directly
- `run(...)` is the repeated fast path inside that program

## Chosen Target

The first proof should build one narrow target:

- platform: CUDA
- backend: NVSHMEM
- dispatcher: simple tile/rank mapping
- scheduler: one static persistent strategy
- kernel lowering: one persistent lowering path

This target is enough to prove the architecture without reopening runtime
strategy choice.

## Program

The logical program should still be the `cuda_nvshmem_event_copy` shape:

- write payload
- signal remote event
- wait remote event
- consume and verify payload

But the authored object is now the orchestrate program itself.

Initial public API proof:

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
      megacu::args().workspace(workspace).signal(ready.release()));

  orch.submit(
      ops::wait_then_check,
      megacu::over(tile),
      megacu::args().wait(ready.acquire()).workspace(workspace));

  orch.run();
}
```

The example must compile without public task descriptors, raw resource ids,
runtime scheduler objects, or string-based module loading.

## Runtime Path

The runtime proof should look like:

1. build the reusable component target set
2. build the concrete orchestrate target
3. call the compiled orchestrate program on two ranks/GPUs
4. verify payload visibility

The runtime should not call CMake/build logic or select scheduler/backend
strategy.

## Planned Paths

- Public API headers:
  - `include/megacu/orchestrator.h`
  - `include/megacu/views.h`
  - `include/megacu/backends/nvshmem.h`
- Build rules:
  - `cmake/MegacuTargets.cmake`
- Internal implementation:
  - `src/dispatcher/`
  - `src/scheduler/`
  - `src/lowering/`
  - `src/platform/cuda/`
  - `src/backends/nvshmem/`
- Proof:
  - `examples/cuda_nvshmem_event_copy/`
  - `tests/build/`
  - `tests/runtime/`

## Acceptance Evidence

Required evidence:

- CMake/build log proving the reusable component targets were built once
- CMake/build log proving the orchestrate target reused those component
  artifacts
- compile-only checks for the orchestrate target ABI
- direct-call smoke test API checks
- generated-code inspection for the built CUDA/NVSHMEM target
- two-rank payload visibility test where environment permits
- explicit skip reason where local NVSHMEM multi-GPU execution is unavailable

Ready-to-implement criteria:

- the public API proof above can be written using only the planned headers;
- the CMake proof can be written using only the two planned CMake functions;
- each generated/lowered record has a single owner from
  `07-dispatcher-scheduler-kernel.md`;
- `run` has no path to build logic or runtime strategy selection.
