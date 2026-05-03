# File Map

This file map describes the intended runtime-linked implementation ownership.
It also marks the current compiler-like files as transitional so code review
does not mistake them for the target design.

## Public Headers

| Path | Intended ownership | Must not own |
| --- | --- | --- |
| `include/megacu/views.h` | Small value views: status, dtype, tensor views, symmetric buffer views, event storage views, backend/session ids. | Scheduling policy, participant mapping, backend algorithms. |
| `include/megacu/runtime_context.h` | Planned runtime call context: launch view, team view, event storage, target capability. | Graph ownership or heap-backed execution plans. |
| `include/megacu/runtime_config.h` | Planned linked capability envelope for `ConfigureTarget` and `OrchTarget`. | Per-tile schedules or materialized metadata sections. |
| `include/megacu/participants.h` | Planned virtual participant annotation API: typed participant refs, role, placement scope, peer policy, progress requirement. | Backend rank assignment or CMake mapping syntax. |
| `include/megacu/platform/cuda.h` | CUDA launch view and CUDA-specific execution constraints. | NVSHMEM PE identity or participant placement. |
| `include/megacu/backends/nvshmem.h` | NVSHMEM team view, backend/session identity, symmetric-resource view helpers. | CUDA launch shape or scheduler ordering. |

The current `include/megacu/program.h` should be removed or rewritten. The
accepted programming surface is direct C++ orchestration plus participant
annotations, not a program builder that records IR.

## Private Runtime Headers

| Path | Intended ownership | Notes |
| --- | --- | --- |
| `include/megacu/detail/component_runtime.h` | Planned common declarations for dispatcher, scheduler, backend, platform, and target runtime entrypoints. | Should expose typed runtime requests and states, not metadata sections. |
| `include/megacu/detail/runtime_validation.h` | Runtime validation declarations used by target runtime and tests. | May remain, but should validate runtime views and capabilities, not metadata headers. |

The following current headers are transitional and should disappear as
implementation dependencies:

| Path | Why transitional |
| --- | --- |
| `include/megacu/detail/program_ir.h` | Stores program-builder records; architecture rejects program IR as the implementation dependency. |
| `include/megacu/detail/materialize.h` | Owns copied IR and materialization; architecture rejects materialization as the normal path. |
| `include/megacu/detail/target_metadata.h` | Defines static dispatch/schedule/kernel/backend sections; architecture requires runtime components instead. |
| `include/megacu/detail/components.h` | Declares section builders; should become runtime component declarations or be removed. |

## Source Directories

| Path | Intended ownership | Current status |
| --- | --- | --- |
| `src/dispatcher/` | General `ConfigureTarget` dispatcher implementation. Input: participant annotations, problem view, runtime context. Output: compact `dispatch_state` with tile/lane/peer work and co-residency constraints. | Currently builds `dispatch_section` from `owned_program_ir`; replace. |
| `src/scheduler/` | Runtime phased/overlap scheduler implementations. Input: `runtime_context`, `dispatch_state`, workload views, linked operator symbols. | Currently builds `schedule_section`; replace with runtime scheduler calls. |
| `src/platform/cuda/` | CUDA validation and launch feasibility checks, including persistent/cooperative constraints for overlap. | Current device check can remain but must grow beyond link anchors. |
| `src/backends/nvshmem/` | NVSHMEM team/resource validation and device-side primitive wrappers. | Current validation can remain; backend metadata builder must be replaced. |
| `src/target/` | Common target-runtime glue: validation sequence, capability checks, status propagation, reusable fast-path helpers. | Currently mostly a link anchor; move reusable example glue here. |
| `src/operators/` | Planned home for reusable operator wrappers if a second example proves common ownership. | `src/lowering/` should not remain as a compiler-stage name. |
| `src/program/` | No target ownership in the corrected architecture. | `src/program/materialize.cc` should be removed with materialization tests. |

## Example Ownership

| Path | Intended ownership |
| --- | --- |
| `examples/cuda_nvshmem/gemm_allreduce/common/` | Problem/workspace types, direct ABI declarations, participant annotation helpers, shared example-only validation. |
| `examples/cuda_nvshmem/gemm_allreduce/golden/` | Expected local GEMM results only. Must not be the implementation called by Megacu targets. |
| `examples/cuda_nvshmem/gemm_allreduce/phased/baseline/` | Pure CUDA/NVSHMEM phased implementation for single-card and two-card runs. |
| `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/` | Megacu phased `OrchTarget`, participant annotations, linked phased operator symbol. |

## Concrete Ownership Rule

The implementation path must not copy facts from one owner to another through
static metadata sections. Each runtime component consumes the values it owns:

```text
OrchTarget
  owns direct ABI + workload types + participant annotations + operator symbols

ConfigureTarget
  owns linked dispatcher + scheduler + platform + backend + target runtime

runtime call
  passes views directly through the components
```
