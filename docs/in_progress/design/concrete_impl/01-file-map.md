# File Map

This document explains the ownership of each current Megacu implementation
file. It is a map for implementation and review, not a stable architecture
contract.

## Public Headers

| Path | Owns | Main types or functions | Must not own |
| --- | --- | --- | --- |
| `include/megacu/views.h` | Plain runtime value views shared by examples, adapters, and orchestrate functions. | `status`, `dtype`, `tensor_view`, `symmetric_buffer_view`, `symmetric_tensor_view`, `event_storage_view`, `backend_id`, `session_id`. | Platform launch policy, scheduler behavior, NVSHMEM semantics beyond opaque ids. |
| `include/megacu/platform/cuda.h` | CUDA-specific runtime launch view and kernel context shape. | `megacu::cuda::launch_view`, `kernel_context`. | Generic program semantics or NVSHMEM team validation. |
| `include/megacu/backends/nvshmem.h` | NVSHMEM-specific runtime team view. | `megacu::nvshmem::team_view`, `ownership`. | CUDA launch behavior, program authoring, scheduler decisions. |
| `include/megacu/program.h` | Public C++ authoring surface for logical programs. | `program_builder`, typed refs for extents/domains/participants/resources/events, `over`, `place`, `remote_event`, `args`, event wait specs. | Backend peer mapping, concrete schedules, CUDA launch shapes, generated source. |

The public headers are intentionally value-oriented. They let a program
describe what exists and what is called, but do not expose dispatcher,
scheduler, lowering, or backend internals.

## Private Detail Headers

| Path | Owns | Main types or functions | Must not own |
| --- | --- | --- | --- |
| `include/megacu/detail/program_ir.h` | The immutable view records produced by `program_builder`. | `program_ir`, `extent_decl`, `domain_decl`, `participant_decl`, `resource_decl`, `event_decl`, `submission_decl`, `event_use`, `arg_binding`. | Runtime ownership, backend lowering, CMake policy. |
| `include/megacu/detail/materialize.h` | Owned copies of program IR and template entrypoint for materialization. | `owned_program_ir`, `materialized_target`, `own`, `materialize_program`, `has_blocking_wait`, `lower_target`. | Component implementation details beyond sequencing. |
| `include/megacu/detail/target_metadata.h` | Concrete metadata sections shared by materialization tests and linked targets. | `target_options`, `target_metadata`, `program_section`, `dispatch_section`, `schedule_section`, `kernel_section`, `backend_section`, `metadata_header`, `serialize`, `validate`. | Large runtime plans or generated platform code. |
| `include/megacu/detail/components.h` | Private component builder declarations. | `build_tiled_compute_comm_dispatch`, `build_static_persistent_schedule`, `build_persistent_stitch_kernel_section`, `build_nvshmem_backend_section`. | Public API. |
| `include/megacu/detail/runtime_validation.h` | Runtime validation function declarations used by example orchestrate paths and tests. | `validate_cuda_launch`, `validate_nvshmem_team`, `validate_nvshmem_symmetric_storage`. | Device kernels or materialized metadata construction. |

The detail headers are currently exposed to tests and example plumbing. They
are not meant to be the user-facing programming model.

## Source Files

| Path | Owns | Current behavior |
| --- | --- | --- |
| `src/program/materialize.cc` | The materialization pipeline driver. | Fills `program_section`, then calls dispatcher, scheduler, kernel lowering, and NVSHMEM backend builders in order. |
| `src/dispatcher/tiled_compute_comm_dispatch.cc` | Dispatch metadata from submissions and participants. | Classifies submissions as compute or communication by matching participant slots against event producer/consumer slots. Creates one work entry per submission and one participant mapping per participant per logical rank. |
| `src/scheduler/static_persistent.cc` | Schedule metadata for phased and co-resident persistent modes. | Turns dispatch entries into ordered schedule entries, records event waits/releases, assigns phase 1 to communication in phased mode, and creates one residency group for co-resident persistent mode. |
| `src/lowering/persistent_stitch.cc` | Kernel-lowering metadata. | Records one kernel symbol per submitted op and launch-shape metadata. Marks overlap schedules as stitched persistent. It does not emit CUDA/C++ source. |
| `src/backends/nvshmem/backend.cc` | NVSHMEM backend metadata. | Records team size, event storage byte requirement, event offsets, symmetric partial-buffer requirement, and multimem capability flag. |
| `src/backends/nvshmem/validation.cc` | NVSHMEM runtime validation. | Checks team envelope and symmetric storage session identity for event storage and partial buffers. |
| `src/platform/cuda/platform.cc` | CUDA component link anchor. | Exposes `megacu_platform_cuda_component()` so the component library has a concrete CUDA platform object. |
| `src/platform/cuda/validation.cc` | CUDA runtime validation. | Checks that `launch.device_ordinal` matches `team.cuda_device_ordinal`. |
| `src/target/runtime.cc` | Target runtime link anchor. | Exposes `megacu_target_runtime_component()`. Current target runtime behavior mostly lives in headers and example glue. |

## Example Files That Exercise The Implementation

| Path | Owns |
| --- | --- |
| `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h` | Logical GEMM+AllReduce programs and direct orchestrate ABI declarations. |
| `examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce_orchestrate_common.h` | Shared runtime validation and dispatch from linked metadata to native CUDA/NVSHMEM entrypoints. |
| `examples/cuda_nvshmem/gemm_allreduce/common/megacu_native_common.h` | Adapter from Megacu runtime views to Megacu-free golden/native CUDA+NVSHMEM views. |
| `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/gemm_allreduce_phased_orchestrate.cc` | Linked phased metadata header and public phased orchestrate function. |
| `examples/cuda_nvshmem/gemm_allreduce/overlap/megacu/gemm_allreduce_overlap_orchestrate.cc` | Linked overlap metadata header and public overlap orchestrate function. |
| `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/megacu_gemm_allreduce_phased.cu` | Current phased Megacu native symbol, routing single-card and multi-card cases to pure CUDA/NVSHMEM implementations. |
| `examples/cuda_nvshmem/gemm_allreduce/overlap/megacu/megacu_gemm_allreduce_overlap.cu` | Current overlap Megacu native symbol, routing single-card and multi-card cases to pure CUDA/NVSHMEM implementations. |

## Ownership Boundary

The intended boundary is:

```text
program.h
  records logical facts
      |
detail/materialize.h + src/program/materialize.cc
  owns copied facts and creates target metadata
      |
src/dispatcher + src/scheduler + src/lowering + src/backends
  build private metadata sections
      |
example orchestrate target
  validates linked metadata and runtime views
      |
native CUDA/NVSHMEM entrypoints
  execute the current numeric implementation
```

The current weak point is that the runtime numeric execution path does not yet
consume the materialized `dispatch_section`, `schedule_section`, or
`kernel_section` payloads. Those sections are produced and tested as metadata,
but the CUDA kernels are still hand-connected through example glue.
