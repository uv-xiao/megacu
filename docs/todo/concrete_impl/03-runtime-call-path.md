# Runtime Call Path

This document traces the intended runtime-linked call path for the
CUDA+NVSHMEM GEMM+AllReduce example. It replaces the old build-time
materialization trace.

## Phased Orchestrate Call

```cpp
cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    workspace, events, launch, team, problem);
```

Concrete call path:

```text
user/test/framework code
  -> cuda_nvshmem_gemm_allreduce_phased_orchestrate(...)
  -> make runtime_context from launch/team/events/capability
  -> target runtime validates capability and common runtime views
  -> example supplies GEMM producer + reduction consumer participant attrs
  -> ConfigureTarget dispatcher maps attrs + problem + team to dispatch_state
       team_n_pes == 1: local tile work, no remote peer work
       team_n_pes == 2: local tile work plus peer reduction work
  -> phased scheduler consumes dispatch_state
       permits reduction tile after matching GEMM tile readiness
       does not require co-resident blocking progress
  -> scheduler calls linked phased operator symbol
  -> operator uses CUDA/NVSHMEM backend primitives as needed
  -> status returns through target runtime
```

## Overlap Orchestrate Call

```cpp
cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    workspace, events, launch, team, problem);
```

Concrete call path:

```text
user/test/framework code
  -> cuda_nvshmem_gemm_allreduce_overlap_orchestrate(...)
  -> make runtime_context from launch/team/events/capability
  -> target runtime validates capability and common runtime views
  -> example supplies GEMM producer + reduction consumer participant attrs
       reduction consumer may block on device-side NVSHMEM waits
  -> ConfigureTarget dispatcher maps attrs + problem + team to dispatch_state
       emits local/peer tile work
       emits co-residency groups for producer/consumer progress
  -> overlap scheduler consumes dispatch_state
       validates co-residency groups against scheduler/operator capability
       asks CUDA platform to validate persistent/cooperative launch feasibility
  -> scheduler calls linked overlap operator symbol
  -> operator keeps compute and communication participants live together
  -> backend device primitives perform NVSHMEM signal/wait/reduction
  -> status returns through target runtime
```

## Distributed Launch Adapters

MPI, `nvshmrun`, torch-distributed, and direct C++ paths all converge before
the orchestrate ABI:

```text
launch adapter
  -> cuda::launch_view
  -> nvshmem::team_view
  -> event_storage_view
  -> direct orchestrate ABI
```

Launch adapters do not choose participant placement. Dispatcher uses
`team_view` and participant annotations to retarget single-card or multi-card
work.

## Validation-Only Smoke Path

Validation-only tests may pass a null stream only when the test explicitly
documents that no CUDA work should launch:

```text
orchestrate call
  -> target/platform/backend validation
  -> dispatcher maps runtime state
  -> scheduler validates progress envelope
  -> returns before native launch because launch.stream is null
```

This path is useful for ABI and validation checks. It is not correctness
evidence for CUDA/NVSHMEM execution.

## Rejected Path

The runtime path must not be:

```text
program_builder
  -> program_ir
  -> materialize_program
  -> dispatch_section/schedule_section/kernel_section/backend_section
  -> linked metadata_header
  -> example glue selects native symbol
```

Those steps describe the transitional implementation rejected by the
architecture. The future concrete implementation must replace them with a
general runtime-linked path.
