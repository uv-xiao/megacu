# Runtime-Linked Device-Native Layer Overview

Status: active redesign after PR #3 review.

This workstream was moved back from `docs/design/` because the previous
implementation-ready design still described a compiler-like path: record a
program IR, materialize owned facts, build dispatch/schedule/kernel/backend
sections, and then validate linked metadata. That is not the Megacu layer we
want.

Megacu is a thin runtime-linked layer:

- CMake links a `ConfigureTarget`: platform, backend, general dispatcher,
  scheduler, and target-runtime components.
- The public C++ surface exposes direct orchestrate functions and typed runtime
  views.
- Dispatcher, scheduler, platform, and backend components run at runtime inside
  the linked orchestrate target.
- The dispatcher is part of the `ConfigureTarget`, but it is not
  workload-specific. It maps annotated virtual participants to runtime ranks,
  lanes, peers, and work ownership using its linked algorithm.
- Kernels/operators are handwritten native implementations linked into the
  target.
- No normal Megacu path builds a program IR, lowers an IR, materializes target
  metadata sections, or generates source.

## Non-Negotiable Correction

The corrected architecture must not include:

- `program_ir` as an implementation dependency;
- `materialize_program`;
- `owned_program_ir`;
- `target_metadata` as the main target contract;
- static `dispatch_section`, `schedule_section`, `kernel_section`, or
  backend-section construction;
- a host materializer executable;
- target lowering as a compiler stage;
- runtime overhead from building intermediate facts before each launch.

The implementation may still expose small typed runtime structs, but only when
they are passed directly to runtime components or kernels.

## Desired Runtime Shape

```text
user/framework/native driver
  calls cuda_nvshmem_gemm_allreduce_*_orchestrate(...)
        |
        v
orchestrate target validates typed views and linked ConfigureTarget capability
        |
        v
runtime dispatcher maps participant annotations plus problem/team to tile,
rank, lane, and peer work
        |
        v
runtime scheduler chooses phased or overlap progress actions for this call
        |
        v
platform/backend adapters validate CUDA/NVSHMEM handles and expose primitives
        |
        v
linked native CUDA/NVSHMEM operator implementation runs
```

This is ordinary linked C++/CUDA. The scheduler and dispatcher are components,
not static compiler passes. Runtime work should be compact enough to disappear
into the same loops and launch setup an expert would write by hand.

## Terms In The Corrected Design

- **ConfigureTarget**: the linked reusable configuration: platform, backend,
  general dispatcher, scheduler, target-runtime helpers, and capability
  envelope. It is selected by CMake and compiled once as normal C++/CUDA code.
- **OrchTarget**: a linked C++/CUDA library exposing one direct orchestrate
  ABI. It depends on one `ConfigureTarget`, supplies workload-specific
  problem/workspace types, virtual-participant annotations, and native operator
  symbols. Example: `cuda_nvshmem_gemm_allreduce_overlap`.
- **Target**: a shorthand when the distinction is not important. When mapping
  or dispatch ownership is discussed, use `ConfigureTarget` or `OrchTarget`
  explicitly.
- **Capability envelope**: small static facts about the linked target:
  platform, backend, dispatcher algorithm, scheduler mode, supported
  dtype/layout/team-size range, and required native symbols. It is not a
  dispatch table or serialized IR.
- **Runtime context**: the handles passed to a call: CUDA launch view, NVSHMEM
  team view, event storage, and the target capability envelope.
- **Task**: a runtime unit of work such as "produce this GEMM tile" or
  "consume this reduction tile". A task is represented by a compact work cursor
  or by native operator code, not by a stored task descriptor table.
- **Event**: a runtime readiness protocol backed by caller-provided storage.
  Events are not looked up by string name. For CUDA+NVSHMEM first slice, the
  event protocol is tile-ready signaling over local or symmetric storage.
- **Virtual participant**: a typed logical actor named by the orchestration
  code before concrete rank, lane, or worker placement is known. Examples:
  compute producer, communication consumer, or mixed participant.
- **Participant attributes**: small workload-supplied runtime annotations on a
  virtual participant, such as role, placement scope, locality, progress
  requirement, communication capability, peer policy, and co-residency
  requirement. They are direct declarations consumed by the dispatcher, not
  stored program IR.
- **Dispatcher**: the `ConfigureTarget` runtime component that maps annotated
  virtual participants plus the current problem/team to work cursors, backend
  peer identity, and local lane/worker placement. It is general to the
  platform/backend/scheduler configuration, not a GEMM+AllReduce-only mapper.
- **Scheduler**: runtime code that decides when linked operators run for this
  call under the target's progress policy.
- **Backend**: runtime code and device helpers for communication semantics.
  For the first backend, that means NVSHMEM team/resource validation and
  device-side signal/wait/reduction primitives.
- **Platform**: runtime code for accelerator launch constraints and errors.
  For the first platform, that means CUDA stream/device/launch validation.

## Responsibility Partition

The architecture needs a hard split between spatial mapping, temporal progress,
resource semantics, and validation. Without that split, dispatcher becomes an
example-specific scheduler or scheduler becomes a hidden distributed mapper.

| Owner | Owns | Does not own |
| --- | --- | --- |
| `OrchTarget` | Direct ABI, workload problem/workspace types, virtual participant annotations, native operator symbols. | Backend rank resolution, launch capability checks, global scheduler policy. |
| `ConfigureTarget` | Linked platform, backend, general dispatcher, scheduler, target runtime, capability envelope. | Workload-specific operator bodies or per-example dispatcher code. |
| Dispatcher | Spatial/topology mapping: participant to rank/lane/peer/work cursor, single-card versus multi-card retargeting, and mapping constraints derived from participant attributes. | Launch ordering, blocking wait progress, CUDA stream/device validation, NVSHMEM primitive implementation. |
| Scheduler | Temporal/progress policy: phased versus overlap ordering, tile readiness consumption, persistent/co-resident progress guard validation, operator invocation sequence. | Backend peer identity, rank mapping, symmetric storage ownership. |
| Backend | Communication resource semantics: NVSHMEM team identity, symmetric storage, device-side signal/wait/reduction primitives. | Participant placement policy or scheduler ordering. |
| Platform | Accelerator execution feasibility: CUDA stream/device, launch shape, cooperative/persistent capability, error conversion. | Backend rank identity or participant mapping. |
| Target runtime | Common validation sequence, status propagation, capability checks. | Owning mapping, progress, or communication algorithms. |

The dispatcher is responsible for retargeting the same `OrchTarget` between
single-card and multi-card runs. For `team_n_pes == 1`, it maps remote peer
policies to local/no-remote work and marks communication participants as local
only. For `team_n_pes > 1`, it maps participant peer policies to backend peers
from `team_view`, validates that the linked backend capability can satisfy the
mapping, and exposes work cursors that scheduler/operator code can consume.

Overlap co-residency is a shared contract:

1. `OrchTarget` marks communication participants as blocking or
   co-resident-progress-required through participant attributes.
2. Dispatcher maps those participants into compatible lane/co-residency groups
   and records the required progress constraints in `dispatch_state`.
3. Scheduler chooses an overlap progress model and rejects the call if the
   dispatch state, operator capability, or launch envelope cannot keep compute
   and communication participants live together.
4. Platform validates CUDA launch feasibility for that progress model.
5. Backend provides the device-side communication primitives; it does not decide
   whether the progress model is legal.

## Directory Scope

This in-progress design owns the next implementation direction for:

- `include/megacu/`
- `src/dispatcher/`
- `src/scheduler/`
- `src/platform/cuda/`
- `src/backends/nvshmem/`
- `src/target/`
- `cmake/MegacuTargets.cmake`
- `examples/cuda_nvshmem/gemm_allreduce/`
- distributed launch and framework adapters.

## Architectural Review Result

Three approaches were considered after rejecting the materializer:

1. **Keep program authoring, but run the collected program at runtime.**
   This keeps too much of the compiler shape and still requires program record
   ownership.
2. **Keep only linked components and direct runtime views.**
   This is the selected direction. It is the thinnest design and matches
   MPK/triton-dist/MegaKittens-style native implementation weight.
3. **Make a generic runtime graph/executor.**
   This is rejected because it adds scheduler and graph overhead to the hot
   path and recreates the weight we are trying to avoid.

The selected direction is option 2.

## Reading Order

1. `00-overview.md`: the runtime-linking correction.
2. `01-programming-surface.md`: what users write in C++.
3. `02-build-link-config.md`: what CMake links and what it must not do.
4. `03-runtime-components.md`: dispatcher, scheduler, platform, backend, and
   target runtime contracts.
5. `04-distributed-runtime.md`: CUDA+NVSHMEM under MPI, torch-distributed, and
   single-process launch.
6. `05-gemm-allreduce-example.md`: phased and overlap GEMM+AllReduce path.
7. `06-implementation-plan.md`: concrete replacement plan for PR #3.
8. `07-verification.md`: evidence required before claiming a working slice.
