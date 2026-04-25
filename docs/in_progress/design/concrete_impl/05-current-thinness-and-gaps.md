# Current Thinness And Gaps

The current implementation is better organized than the placeholder component
anchor, but it is still thin. This document makes that explicit so follow-up
work can target the real gaps instead of only adding more labels.

## What Is Concrete Today

- `include/megacu/program.h` has a real authoring API that records extents,
  domains, participants, resources, events, submissions, args, and event uses.
- `materialize_program<Program>` creates owned IR and target metadata.
- Component source files exist under the intended ownership directories:
  `src/program`, `src/dispatcher`, `src/scheduler`, `src/lowering`,
  `src/platform/cuda`, `src/backends/nvshmem`, and `src/target`.
- Dispatcher metadata records work entries and virtual participant/backend peer
  mappings.
- Scheduler metadata records phased versus co-resident persistent entries,
  event waits/releases, residency groups, and a CUDA residency envelope.
- Kernel lowering metadata records op symbols and launch shape. It does not
  generate code.
- NVSHMEM backend metadata records event layout and symmetric-buffer
  requirements.
- Runtime validation checks metadata headers, team envelope, CUDA device
  consistency, workspace pointers, symmetric-storage session identity, and f32
  numeric storage size.
- CUDA+NVSHMEM GEMM+AllReduce has direct orchestrate functions for phased and
  overlap variants.
- Tests compile descriptors, materialize metadata, inspect component sections,
  validate runtime adapters, link metadata symbols, smoke direct orchestrate
  calls, and run CUDA numeric correctness.

## What Is Still Too Thin

### Metadata Does Not Fully Drive Runtime

The runtime path validates a linked `metadata_header`, but it does not consume
the full `target_metadata` object. In particular, native CUDA execution does
not currently read:

- `dispatch.work`;
- `dispatch.participants`;
- `schedule.entries`;
- `schedule.residency_groups`;
- `kernel.symbols`;
- `backend.events`.

The current runtime still calls a hand-written native symbol selected by
progress mode.

### Dispatcher Does Not Expand Real Tiles

The dispatcher currently creates one work entry per submission. A real tiled
GEMM+AllReduce dispatcher should materialize work at tile granularity:

```text
for output tile:
  create compute work for producer participant
  create communication work for consumer participant
  map logical ranks to backend peers
```

The current `tile = {0, 0}` placeholder is inspectable but not enough for
runtime scheduling.

### Phased Scheduling Still Has False Dependency Risk

The phased scheduler marks communication entries as phase 1. That is useful for
showing a phased path, but it can imply a whole-compute-before-communication
barrier. The design goal is stricter:

```text
GEMM tile is ready -> matching AR tile may run
```

The phased path should eventually permit tile-level readiness without requiring
compute and communication workers to be co-resident.

### Overlap Guard Is Metadata-Only

The overlap materialization rejects blocking waits without a co-resident guard.
That is useful. The runtime must also prove that the actual CUDA launch uses
the required residency group and that compute/communication workers are live
together when communication can block.

### Kernel Lowering Lacks Symbol Validation

Kernel lowering records op names and symbol ids. It does not yet verify that a
selected backend target provides all required symbols with compatible roles,
argument layouts, and launch requirements.

### CUDA Platform Validation Is Minimal

Current CUDA validation checks only device ordinal consistency. The first
complete implementation should also validate:

- stream presence when a launch is required;
- residency envelope feasibility;
- cooperative launch requirement if enabled;
- thread/block limits;
- device capability requirements;
- single-card versus multi-card capability selection.

### NVSHMEM Backend Metadata Does Not Select Device Primitives

The backend records event storage and symmetric buffer requirements. It does
not yet drive a reusable device-side API for waits, puts/gets, signals, or
multimem-style reductions.

### Target Runtime Is Mostly Example Glue

`src/target/runtime.cc` is currently only a link anchor. Shared target runtime
behavior should move out of
`examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_orchestrate_common.h`
when the behavior becomes reusable across examples.

### Golden, Baseline, And Megacu Paths Are Still Blurry

The native CUDA/NVSHMEM implementation under `golden/` is currently also used
by Megacu native entrypoints. This is acceptable as a temporary implementation
step, but the final example organization should make these roles visibly
separate:

```text
golden:
  local expected results

phased/baseline and overlap/baseline:
  pure CUDA/NVSHMEM implementations

phased/megacu and overlap/megacu:
  Megacu-authored and Megacu-scheduled implementations
```

## Next Implementation Targets

The next useful implementation work should focus on these concrete milestones:

1. Make dispatcher output tile-granular for GEMM+AllReduce.
2. Make phased schedule consume tile readiness without false whole-program
   dependencies.
3. Make overlap runtime launch enforce the co-resident progress guard.
4. Move shared runtime target behavior from example common code into
   `src/target`.
5. Add lowering validation that required op symbols exist and match expected
   roles.
6. Separate golden, baseline, and Megacu native paths in code, not just
   directory names.
7. Make tests assert that runtime execution consumes dispatch/schedule/lowering
   payloads instead of bypassing them.

## Non-Goals For This Documentation Slice

This documentation does not claim:

- performance parity with MPK, Triton-Distributed, or handwritten kernels;
- zero-overhead runtime execution;
- complete multi-backend retargetability;
- complete CUDA+NVSHMEM production readiness.

Those claims require stronger implementation and measurement evidence than the
current PR has.
