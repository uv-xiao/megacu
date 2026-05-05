# Component Contracts

This document states the concrete runtime-linked contracts for implementation.
The contracts mirror the architecture responsibility partition.

## OrchTarget

Files:

- `examples/cuda_nvshmem/gemm_allreduce/*/megacu/*_orchestrate.cc`
- `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h`

Contract:

- exposes one direct ABI per target;
- defines workload problem and workspace views;
- declares typed virtual participants and `participant_attrs`;
- passes runtime views into target runtime;
- calls dispatcher and scheduler in visible order;
- passes linked native operator symbols to scheduler.

Must not:

- build or own program IR;
- contain backend PE mapping tables;
- call golden functions as production Megacu implementation;
- hide runtime state in an untyped payload blob.

## Dispatcher

Files:

- planned `include/megacu/detail/component_runtime.h`
- planned `src/dispatcher/annotated_runtime.cc`

Contract:

- input: `dispatch_request` with runtime context, problem view, and participant
  bindings;
- output: compact `dispatch_state`;
- retargets one `OrchTarget` between `team_n_pes == 1` and multi-card teams;
- maps participants to local lanes/workers and backend peers;
- emits tile work cursors;
- rejects unsupported participant annotation combinations for the linked
  `ConfigureTarget`.

Must not:

- choose launch ordering;
- decide whether blocking progress is globally legal;
- validate CUDA stream/device;
- implement NVSHMEM primitives;
- specialize to GEMM+AllReduce.

## Scheduler

Files:

- planned `src/scheduler/static_runtime.cc`

Contract:

- input: runtime context, dispatch state, workload views, linked operator
  symbols;
- phased mode consumes tile readiness without introducing false whole-program
  dependencies when tile readiness is available;
- invokes native operator symbols directly.

Must not:

- recompute rank or backend peer mapping;
- own symmetric storage identity;
- perform numeric correctness checks;
- build schedule sections.

## CUDA Platform

Files:

- `include/megacu/platform/cuda.h`
- `src/platform/cuda/validation.cc`

Contract:

- validates device identity and stream requirements;
- validates launch shape for scheduler requirements;
- converts CUDA failures into `megacu::status`.

Must not:

- own NVSHMEM PE identity;
- decide participant placement;
- choose scheduler mode.

## NVSHMEM Backend

Files:

- `include/megacu/backends/nvshmem.h`
- `src/backends/nvshmem/validation.cc`
- planned `src/backends/nvshmem/runtime.cc`

Contract:

- validates team size, PE identity, backend/session identity, and symmetric
  resources;
- distinguishes single-card local execution from multi-card remote execution;
- exposes device-side primitive wrappers for signal, wait, get/put, and
  reduction operations used by native operators.

Must not:

- choose single-card versus multi-card participant mapping;
- choose scheduler ordering;
- own CUDA launch feasibility.

## Target Runtime

Files:

- planned `src/target/runtime.cc`

Contract:

- owns common validation sequencing;
- checks target capability envelope;
- calls platform/backend validation in the correct order;
- propagates `megacu::status`;
- keeps reusable target glue out of example-only headers.

Must not:

- own dispatcher or scheduler algorithms;
- materialize IR;
- validate static metadata sections as the main target contract.

## Transitional Components To Remove

The following contracts are explicitly rejected as final implementation
contracts:

- `program_builder` producing `program_ir`;
- `materialize_program` producing `target_metadata`;
- `build_tiled_compute_comm_dispatch` producing `dispatch_section`;
- `build_static_static_schedule` producing `schedule_section`;
- `build_static_stitch_kernel_section` producing `kernel_section`;
- `build_nvshmem_backend_section` producing backend metadata.
