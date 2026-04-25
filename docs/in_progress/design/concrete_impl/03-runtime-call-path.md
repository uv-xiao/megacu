# Runtime Call Path

This document traces how a Megacu-based example runs with the current
implementation. It focuses on the CUDA+NVSHMEM GEMM+AllReduce example because
that is the first working path in PR #3.

## Build-Time Metadata Path

The metadata path is exercised by build tests and component tests.

```text
gemm_allreduce_phased_program::describe(program_builder)
        |
        v
program_builder records:
  extents: m_tiles, n_tiles, team_size
  domains: output_tile, rank
  participants: compute, reduce
  resources: workspace, events
  event: partial_ready
  submissions: gemm_tile_produce, allreduce_tile_consume
        |
        v
materialize_program<Program>(target_options)
        |
        v
own(program_ir)
        |
        v
lower_target(owned_program_ir, target_options)
        |
        +--> build_tiled_compute_comm_dispatch(...)
        +--> build_static_persistent_schedule(...)
        +--> build_persistent_stitch_kernel_section(...)
        +--> build_nvshmem_backend_section(...)
        |
        v
target_metadata
        |
        v
serialize(target_metadata) or inspect metadata sections in tests
```

Concrete callers:

- `tests/build/compile_gemm_allreduce_descriptor.cc` checks the authored
  descriptor compiles.
- `tests/build/materialize_gemm_allreduce.cc` materializes phased and overlap
  programs and checks JSON/metadata facts.
- `tests/build/component_metadata_contracts.cc` inspects dispatch, schedule,
  lowering, and backend sections.

## Runtime Direct Call Path

The runtime path starts when code calls the direct orchestrate ABI declared in
`examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h`.

Phased:

```cpp
cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    workspace, events, launch, team, problem);
```

Overlap:

```cpp
cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    workspace, events, launch, team, problem);
```

The current call path is:

```text
user/test code
  calls cuda_nvshmem_gemm_allreduce_*_orchestrate(...)
        |
        v
examples/.../*/megacu/gemm_allreduce_*_orchestrate.cc
  passes arguments plus linked metadata_header
        |
        v
gemm_ar_detail::orchestrate_impl(...)
        |
        +--> megacu::detail::validate(metadata_header)
        |
        +--> validate_common(...)
        |       |
        |       +--> validate_nvshmem_team(team, max_supported_team_size = 2)
        |       +--> validate_cuda_launch(launch, team)
        |       +--> check workspace pointers
        |       +--> validate_nvshmem_symmetric_storage(events, partial, team)
        |
        +--> run_numeric(...)
                |
                +--> if launch.stream == nullptr:
                |       return ok after validation only
                |
                +--> validate_numeric_f32(workspace, problem)
                |
                +--> if metadata.progress == phased:
                |       megacu_cuda_gemm_allreduce_phased_f32(...)
                |
                +--> otherwise:
                        megacu_cuda_gemm_allreduce_overlap_f32(...)
```

## Native Symbol Path

The native Megacu symbols live in the variant directories:

```text
phased/megacu/megacu_gemm_allreduce_phased.cu
overlap/megacu/megacu_gemm_allreduce_overlap.cu
```

Each symbol adapts Megacu views to the current pure CUDA/NVSHMEM implementation
views through `gemm_ar_native` helpers:

```text
megacu_cuda_gemm_allreduce_phased_f32(...)
        |
        +--> if team.team_n_pes <= 1:
        |       golden_phased_single_card_gemm_allreduce_f32(...)
        |
        +--> else:
                golden_phased_multi_card_gemm_allreduce_f32(...)

megacu_cuda_gemm_allreduce_overlap_f32(...)
        |
        +--> if team.team_n_pes <= 1:
        |       golden_overlap_single_card_gemm_allreduce_f32(...)
        |
        +--> else:
                golden_overlap_multi_card_gemm_allreduce_f32(...)
```

The native CUDA/NVSHMEM implementation lives in:

```text
examples/cuda_nvshmem/gemm_allreduce/golden/golden_gemm_allreduce.cu
```

Despite the `golden_` prefix, these functions currently serve two roles:

1. Megacu-free golden/baseline execution.
2. The low-level implementation called by Megacu native entrypoints.

This dual use is a known cleanup target because it can blur the separation
between golden, baseline, and Megacu implementation paths.

## Validation-Only Runtime Path

If `launch.stream == nullptr`, `run_numeric` returns `ok` after validating
metadata and runtime views. This path is used by direct ABI smoke tests to
exercise validation without launching CUDA work.

```text
direct_orchestrate_smoke
  -> orchestrate function
  -> metadata validation
  -> CUDA/NVSHMEM/team/workspace validation
  -> no CUDA launch because stream is null
```

## Numeric Runtime Path

If `launch.stream != nullptr`, the runtime path validates f32 storage and calls
the native CUDA/NVSHMEM symbol. The single-card tests use CUDA kernels without
NVSHMEM team launch. The two-rank smoke path uses `nvshmrun` when
`MEGACU_ENABLE_NVSHMEM_TESTS` is enabled.

```text
cuda_gemm_allreduce_correctness
  -> allocate CUDA storage
  -> call phased orchestrate
  -> call overlap orchestrate
  -> compare outputs against expected values

nvshmem_two_rank_smoke_*    optional build mode
  -> nvshmrun -np 2
  -> call golden or Megacu mode
  -> use device-side NVSHMEM path when available
```

## Important Current Mismatch

The metadata path and runtime path are connected by the linked
`metadata_header` progress mode and table counts. The runtime path does not yet
consume the full materialized sections:

- `dispatch_section`;
- `schedule_section`;
- `kernel_section`;
- `backend_section`.

Those sections are inspected by tests, but the CUDA launch path still calls a
hand-written native function. Closing that gap is necessary before claiming the
implementation is a complete Megacu execution pipeline.
