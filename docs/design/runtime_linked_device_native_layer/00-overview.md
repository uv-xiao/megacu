# Runtime-Linked Device-Native Layer Overview

Status: implemented PR #4 runtime-linked slice.

This design replaces the rejected compiler-like path: record a program IR,
materialize owned facts, build dispatch/schedule/kernel/backend sections, and
then validate linked metadata. That is not the Megacu layer we want.

Megacu is a thin runtime-linked layer:

- CMake links a `ConfigureTarget`: platform, backend, general dispatcher,
  scheduler, and target-runtime components.
- The public C++ surface exposes target orchestrate entries that are called
  from host code with a driver plus target-specific arguments. Target arguments
  should look like CUDA-kernel arguments: pointers, scalar values, and small
  trivially copyable descriptors when needed.
- Dispatcher, scheduler, platform, and backend components run at runtime inside
  the linked orchestrate target.
- The dispatcher is part of the `ConfigureTarget`, but it is not
  workload-specific. It maps submitted task/operator calls plus
  component-provided attributes to runtime ranks, lanes, peers, and work
  ownership using its linked algorithm.
- Kernels/operators are handwritten native implementations linked into the
  target. Operators use direct native signatures.
- No normal Megacu path builds a program IR, lowers an IR, materializes target
  metadata sections, or generates source.

## PR #4 Split Scope

PR #4 is an architecture-redesign PR, not the full concrete implementation PR.
It includes one tiny but mighty proof example that is allowed to be
problem-specific and does not need to prove reusable generality.

The broad concrete implementation notes moved to
`docs/todo/concrete_impl/`. That TODO workstream owns the later requirement to
turn the architecture into general reusable components. PR #4's tiny proof is
phased only and must cover 1-host/1-device and 1-host/2-device. In this
architecture workstream, any problem-specific shortcut in the tiny proof
example must be called out as example-local and must not be promoted into
public Megacu API or shared component contracts.

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
  calls gemm_allreduce_phased_orchestrate(driver, a, b, partial, out, m, n, k)
        |
        v
target runtime sequences the thin linked components
        |
        v
programmer-authored orchestration submits operator calls with raw arguments,
explicit dependency attributes, and other component-provided attributes
        |
        v
runtime dispatcher maps submitted work and attributes to local/peer work
        |
        v
runtime scheduler consumes explicit dependency attributes and runs ASAP progress for PR #4
        |
        v
platform/backend adapters expose execution facts and primitives
        |
        v
linked native CUDA/NVSHMEM operator implementation runs
```

This is ordinary linked C++/CUDA. The scheduler and dispatcher are components,
not static compiler passes. Runtime work should be compact enough to disappear
into the same loops and launch setup an expert would write by hand.

In this design, "one megakernel" means one Megacu target operation from the host
caller and one device-side Megacu running loop for PR #4. That running loop may
invoke multiple linked CUDA/NVSHMEM operator kernels internally. The host code
sees one orchestrate call and one driver-owned asynchronous execution surface;
the Megacu target owns the internal dispatch and readiness sequence.

## Terms In The Corrected Design

- **ConfigureTarget**: the linked reusable configuration: platform, backend,
  general dispatcher, scheduler, target-runtime helpers, and capability
  envelope. It is selected by CMake and compiled once as normal C++/CUDA code.
- **OrchTarget**: a linked C++/CUDA library exposing one orchestrate entry with
  a target-specific host-call signature. It depends on one `ConfigureTarget`,
  supplies workload-specific orchestration code, target argument descriptions,
  attributes, and native operator symbols. Example: `gemm_allreduce_phased`.
- **Target**: a shorthand when the distinction is not important. When mapping
  or dispatch ownership is discussed, use `ConfigureTarget` or `OrchTarget`
  explicitly.
- **Capability envelope**: small static facts about the linked target:
  platform, backend, dispatcher algorithm, scheduler mode, supported
  dtype/layout/team-size range, and required native symbols. It is not a
  dispatch table or serialized IR.
- **Driver**: a platform/backend execution object passed by host code to the
  orchestrate entry. For CUDA+NVSHMEM it owns asynchronous execution resources
  such as the CUDA stream/device context, NVSHMEM PE/team identity, team size,
  and distributed launch facts. The driver is not a
  target argument; it is the execution authority for launching and
  synchronizing the target megakernel.
- **Megakernel**: one linked Megacu target operation invoked by one host
  orchestrate call. It contains a common device-side running loop. The linked
  dispatcher chooses work, the linked scheduler chooses ready tasks, and the
  operator table invokes target operator kernels inside that one operation.
- **Target arguments**: target-specific values passed after the driver. They
  should be shaped like CUDA kernel arguments: device pointers, POD scalars,
  and small descriptor structs. Megacu should not wrap them into a generic
  tensor/scalar/resource bag on the public path.
- **Task**: a submitted operator call with raw pointer/scalar/descriptor
  arguments, explicit dependency attributes, and other component-provided
  attributes. A task is runtime execution state for one call, not a stored
  compiler IR and not a read/write/inout argument abstraction.
- **Sync task**: a task with no operator body. It represents readiness only:
  join, split, coordinate reshape, barrier-like progress, or data-dependent
  triggering. It is Megacu's native Event Tensor middle node, lowered by the
  scheduler/platform/backend into internal counters, waits, signals, or queue
  pushes.
- **Attribute**: typed metadata supplied by a component and attached by
  orchestration code to scopes or operator submissions. Megacu core owns only
  the composition and lookup interface; dispatcher, scheduler, platform, and
  backend components own their vocabularies.
- **Event**: a concrete readiness protocol used by components when an explicit
  dependency attribute requires device-visible signaling. Events are not looked
  up by tensor name or string name.
- **Dispatcher**: the `ConfigureTarget` runtime component that maps submitted
  tasks, attributes, and driver resources to work cursors, backend peer
  identity, and local lane/worker placement. It is general to the
  platform/backend/scheduler configuration, not a GEMM+AllReduce-only mapper.
- **Scheduler**: runtime code that decides when linked operators run for this
  call under the target's progress policy.
- **Backend**: runtime code and device helpers for communication semantics.
  For PR #4, that means simple NVSHMEM team facts and device-side
  signal/wait/reduction primitives, not a heavy validator.
- **Platform**: runtime code for accelerator launch facts and errors.
  For PR #4, that means simple CUDA stream/device facts and native-operator
  error conversion, not deep launch validation.

## Responsibility Partition

The architecture needs a hard split between spatial mapping, temporal progress,
resource semantics, and validation. Without that split, dispatcher becomes an
example-specific scheduler or scheduler becomes a hidden distributed mapper.

| Owner | Owns | Does not own |
| --- | --- | --- |
| `OrchTarget` | Orchestrate entry, target-specific argument shape, workload orchestration code, operator signatures, event tensor declarations, component attributes, native operator symbols. | Backend rank resolution, launch capability checks, global scheduler policy. |
| `ConfigureTarget` | Linked platform, backend, general dispatcher, scheduler, target runtime, capability envelope. | Workload-specific operator bodies or per-example dispatcher code. |
| Dispatcher | Spatial/topology mapping: submitted work to rank/lane/peer/work cursor, single-device versus two-device retargeting, mapping constraints derived from dispatcher-owned attributes. | Launch ordering, dependency execution, CUDA stream/device validation, NVSHMEM primitive implementation. |
| Scheduler | Temporal/progress policy: explicit dependency-attribute consumption, ASAP task readiness in PR #4, operator invocation sequence. | Backend peer identity, rank mapping, symmetric storage ownership, tensor-access dependency inference. |
| Backend | Communication resource semantics: NVSHMEM team identity, symmetric storage, and lowering of event tensor handlers to device-side signal/wait/reduction primitives. | Work placement policy or scheduler ordering. |
| Platform | Accelerator execution feasibility: CUDA stream/device resources, launch shape, error conversion. | Backend rank identity or work mapping. |
| Target runtime | Common component sequencing and status propagation. | Owning mapping, progress, communication algorithms, or heavy validation. |

The driver must stay narrow. It includes only resources that the host execution
environment owns and that platform/backend components need for asynchronous and
distributed execution: stream/device or context, backend PE/team identity,
team size, and distributed launch facts. It does not include target arguments,
target argument storage, linked
component sets, capability envelopes, operator argument packs, or target-local
scratch/events. Those belong to the host call arguments, the linked target, or
the target implementation. This boundary keeps `driver` from becoming the
generic runtime bag that the design rejected.

The dispatcher is responsible for retargeting the same `OrchTarget` between
1-host/1-device and 1-host/2-device runs in PR #4. For one device, it maps the
phased work locally. For two devices, it maps the same submitted task structure
to local plus peer reduction work using backend resources from the driver. The
target argument shape and orchestration code stay the same.

This is the destination contract for the general implementation. PR #4's tiny
proof may use a problem-specific mapper if that keeps the proof focused on the
architecture correction. Such a mapper is evidence for the call shape only; it
is not accepted as the reusable dispatcher contract.

Overlap co-residency is future work. For PR #4, communication in the phased
example should be represented by one fused native mega-kernel when that keeps
the proof small. The design should not force several standalone communication
kernels just to exercise dispatcher and scheduler complexity.

## Directory Scope

This design owns the architecture direction for:

- `include/megacu/`
- `src/dispatcher/`
- `src/scheduler/`
- `src/platform/cuda/`
- `src/backends/nvshmem/`
- `src/target/`
- `cmake/MegacuTargets.cmake`
- `examples/cuda_nvshmem/gemm_allreduce/`
- single-host CUDA+NVSHMEM launch resources for 1-device and 2-device PR #4
  evidence. MPI and torch-distributed adapters are future work.

## Architectural Review Result

Three approaches were considered after rejecting the materializer:

1. **Keep program authoring, but run the collected program at runtime.**
   This keeps too much of the compiler shape and still requires program record
   ownership.
2. **Keep only linked components, direct target argument signatures, and component
   resources behind a driver.**
   This is the selected direction. It keeps programmer code close to ordinary
   host-launched CUDA/C++ while preserving normal linked execution.
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
5. `04-distributed-runtime.md`: PR #4 single-host 1-device/2-device runtime
   resources and future distributed adapters.
6. `05-gemm-allreduce-example.md`: phased-only tiny GEMM+AllReduce proof.
7. `06-implementation-plan.md`: PR #4 architecture repair plan and scope
   boundary.
8. `07-verification.md`: evidence required before claiming a working slice.
9. `08-program-compile-execute-flow.md`: intuitive program, compile, and
   execution flow for CUDA+NVSHMEM.
