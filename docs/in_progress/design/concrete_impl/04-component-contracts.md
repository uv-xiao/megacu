# Component Contracts

This document states what each current implementation component promises today.
It also states the boundary that the next implementation work should preserve.

## Program Authoring

Files:

- `include/megacu/program.h`
- `include/megacu/detail/program_ir.h`

Contract:

- Program authors call `Program::describe(megacu::program_builder&)`.
- The builder assigns compact `slot_index` values in declaration order.
- Typed refs keep user code from passing arbitrary integers at the public
  authoring surface.
- The output is a non-owning `program_ir` view into builder-owned vectors.

Current concrete records:

- extents;
- domains with extent slots;
- participants;
- resources with C++ view type identity;
- remote events with from/to participant slots, storage resource slot, and
  memory scope;
- submissions with op name, work domain, participant placement, args, and event
  uses.

Boundary:

- `program_builder` records logical facts only.
- It does not map participants to ranks.
- It does not choose CUDA launch shape.
- It does not know whether a backend is NVSHMEM or something else.

## Materialization

Files:

- `include/megacu/detail/materialize.h`
- `src/program/materialize.cc`

Contract:

- `own(program_ir)` copies builder views into `owned_program_ir` so later stages
  do not depend on the lifetime of `program_builder`.
- `materialize_program<Program>(target_options)` calls `Program::describe`,
  owns the IR, checks the co-resident blocking-wait guard, and lowers the
  target.
- `lower_target` executes the first-slice component order:

```text
program counts
  -> dispatcher
  -> scheduler
  -> kernel lowering metadata
  -> NVSHMEM backend metadata
```

Boundary:

- Materialization can reject impossible metadata contracts.
- It should not emit source files.
- It should not perform runtime CUDA/NVSHMEM validation.

Current weakness:

- The linked example metadata header is still manually written in orchestrate
  source files rather than derived from the complete `target_metadata` object.

## Dispatcher

File:

- `src/dispatcher/tiled_compute_comm_dispatch.cc`

Contract:

- Input: owned program IR and target options.
- Output: `dispatch_section`.
- Classifies work as compute when the submission participant is an event
  producer and communication when the participant is an event consumer.
- Creates one `dispatch_entry` per submission.
- Creates participant/backend peer mappings for each participant and each
  logical rank in `target_options.team_size`.

Current concrete output:

```text
dispatch.work:
  tile = {0, 0}
  role = compute or comm
  participant_slot = submission.participant_slot
  logical_rank = 0
  worker_index = submission order

dispatch.participants:
  participant_slot
  logical_rank
  backend_peer = logical_rank
```

Boundary:

- Dispatcher owns virtual participant to logical rank/backend peer mapping.
- CMake should not contain ad hoc participant-to-rank mapping syntax.

Current weakness:

- The dispatcher is still symbolic. It does not expand real tile coordinates
  from runtime problem sizes, and it creates only one work entry per submission
  instead of one work entry per materialized tile.

## Scheduler

File:

- `src/scheduler/static_persistent.cc`

Contract:

- Input: owned program IR, dispatch section, and target options.
- Output: `schedule_section`.
- For each dispatch work entry, creates one schedule entry with action,
  order, event wait/release slots, phase, role, and optional residency group.
- In phased mode, communication work is assigned `phase = 1`.
- In co-resident persistent mode, all work goes into residency group 0.
- Blocking device waits require `co_resident_guard` in
  `materialize_program`.

Current concrete output:

```text
phased:
  compute entries phase 0
  comm entries phase 1
  no residency group

co_resident_persistent:
  compute and comm entries residency_group 0
  residency group records required compute and comm workers
  residency envelope records grid blocks and threads per block
```

Boundary:

- Scheduler owns readiness, phase, and co-residency facts.
- Scheduler should make progress guarantees inspectable before lowering.

Current weakness:

- The phased scheduler still imposes coarse phase labels instead of proving
  per-tile readiness without false whole-program dependencies.
- The runtime CUDA path does not consume schedule entries yet.

## Kernel Lowering

File:

- `src/lowering/persistent_stitch.cc`

Contract:

- Input: owned program IR and schedule section.
- Output: `kernel_section`.
- Records one `kernel_symbol` per unique submitted op.
- Records launch shape metadata from the schedule residency envelope.
- Marks `stitched_persistent = true` for co-resident persistent schedules.

Boundary:

- Kernel lowering links named high-level ops to existing implementation roles.
- It must not generate CUDA or C++ source.

Current weakness:

- Symbols are inspectable metadata only. Missing-symbol validation and runtime
  launch payload binding are not complete.

## CUDA Platform Adapter

Files:

- `include/megacu/platform/cuda.h`
- `src/platform/cuda/platform.cc`
- `src/platform/cuda/validation.cc`

Contract:

- `launch_view` carries a CUDA stream pointer and device ordinal.
- `validate_cuda_launch` checks that the launch device matches the NVSHMEM
  team's CUDA device.
- `megacu_platform_cuda_component()` proves the CUDA platform component is
  linked into the component library.

Boundary:

- CUDA platform code validates CUDA launch resources and records CUDA-specific
  launch context shape.
- It should not own NVSHMEM team semantics or logical program facts.

Current weakness:

- Validation is minimal. It does not yet check cooperative launch
  requirements, persistent residency feasibility, block limits, or stream
  ownership.

## NVSHMEM Backend Adapter

Files:

- `include/megacu/backends/nvshmem.h`
- `src/backends/nvshmem/backend.cc`
- `src/backends/nvshmem/validation.cc`

Contract:

- `team_view` carries opaque team handle, PE identity, CUDA device identity,
  backend id, and session id.
- `build_nvshmem_backend_section` records team size, event-storage byte
  requirements, event offsets, symmetric partial-buffer requirement, and a
  current `may_use_multimem_reduce` flag.
- `validate_nvshmem_team` validates the team envelope for the current maximum
  supported team size.
- `validate_nvshmem_symmetric_storage` checks event and partial buffers against
  team backend/session identity.

Boundary:

- Backend adapter owns native backend capability and runtime resource
  validation.
- It should not own logical program authoring or CUDA launch shape.

Current weakness:

- Backend metadata does not yet drive the device-side communication primitive
  selection at runtime.

## Target Runtime

Files:

- `include/megacu/detail/target_metadata.h`
- `src/target/runtime.cc`
- example orchestrate files under `examples/cuda_nvshmem/gemm_allreduce/*/megacu/`

Contract:

- `metadata_header` is the linked target validation header.
- `validate(metadata_header)` checks magic, version, size, checksum, and
  required table counts.
- Orchestrate functions validate the linked header before runtime launch.

Boundary:

- Target runtime should connect compiled target metadata to a direct ABI.
- It should not hide dynamic graph building behind the call.

Current weakness:

- `src/target/runtime.cc` is only a link anchor today.
- Most target runtime behavior lives in headers and example-specific common
  code. This should be made more reusable.
