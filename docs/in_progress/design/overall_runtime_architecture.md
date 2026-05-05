# Overall Runtime Architecture

Status: active design draft for the general runtime-linked components PR.
Promote the accepted parts into `docs/design/` only during PR closeout.

## Core Shape

Megacu has one host-called orchestrate function per target:

```cpp
megacu::status target_orchestrate(driver, raw_target_args...);
```

The orchestrate function builds or seeds one call-owned orchestration frame and
launches one mega-kernel through the linked runtime.

```text
orchestration_frame
  raw target args
  operator bindings
  tasks[]
  event_tensors[]
  linked component config
```

Normal users do not call `scheduler.run`, do not program waits/signals inside
operators, and do not pass task arguments through a Megacu-specific argument
wrapper.
Operator arguments remain raw CUDA-kernel-like values: pointers, scalars, and
small descriptors.

## Task And Event Model

Tasks and event tensors are separate orchestration objects. Both use attrs, but
their attrs describe different things.

```cpp
struct task_record {
  task_ref id;
  operator_ref op;    // present for submit, empty for sync
  raw_arg_view args;  // operator task only
  attr_set attrs;
};

struct event_tensor_record {
  event_ref id;
  attr_set attrs;
};
```

`submit(...)` creates an operator task and injects the builtin
`task::operator_task{}` attr. `sync(...)` creates a sync-only task and injects
the builtin `task::sync_task{}` attr. Users do not provide task-kind attrs
directly in this PR.

Event tensor object attrs describe the event object:

```text
event_tensor::shape{...}
event_tensor::wait_count{...}
event_tensor::scope{...}
cuda_nvshmem::event_tensor::symmetric_storage{...}
```

Task attrs describe operations on event tensors:

```text
event_tensor::notify(e)
event_tensor::wait(e)     // sync-only task completion condition
event_tensor::trigger(e)  // dynamic scheduling candidate
```

Sync-only tasks are first-class tasks. They are used for readiness work such as
joining, splitting, reshaping, barrier-like progress, or triggering. They go
through the same runtime issue path as operator tasks, but their dispatcher
mapping can be trivial and their behavior is handled by the EventTensor
component.

This API follows the Event Tensor paper's core model: an event tensor element
is a counter-shaped event, producers notify it, and sync-only tasks can wait on
it. Megacu does not copy the paper's finer-grained task-splitting model.
Operator/task is the sync level. If a target needs finer sync, it should submit
smaller operator tasks. Operator kernels still do not call `notify()` or
`wait()` directly; the runtime loop and linked EventTensor component lower the
attrs around operator execution or sync-only task handling.

The common EventTensor API is decoupled from platform/backend lowering. Common
attrs describe logical event tensors and operations. Platform/backend attrs add
storage, memory-scope, rank/team, and communication facts required to lower the
same logical operations on a target.

Megacu EventTensor should be understood as lower-level than, not less
expressive than, the original Event Tensor paper. Paper-style
producer-event-consumer designs can be written in Megacu by submitting operator
tasks at the desired granularity and connecting them through sync-only
EventTensor tasks. The paper's compiler can generate that structure from a
compact tiled graph; Megacu currently exposes the explicit task construction
surface. The tradeoff is authoring/generator burden and lowering quality, not a
semantic inability to represent the design.

## Linked Components

A target links a configuration:

```text
ConfigureTarget<
  runtime::device_persistent<
    runtime::execution::seeded_orch,
    runtime::loop::block_tile>,
  dispatcher::rank_aware_tile_grid,
  scheduler::explicit_asap,
  event_tensor::counter_tensor,
  platform::cuda,
  backend::nvshmem>
```

Ownership:

| Component | Owns | Does not own |
| --- | --- | --- |
| Orchestrate function | Target-specific raw arguments, operator submissions, event tensor declarations, attrs. | Scheduler loop, backend rank setup, fallback dependency inference. |
| Runtime | Runtime execution model and runtime loop: task arena publication/sealing plus device-side issue policy. | Dependency semantics, spatial mapping, event semantics. |
| Runtime execution model | Where/when task and EventTensor records are built, published, and sealed. | Loop discipline, dependency semantics, spatial mapping. |
| Runtime loop | Device-side mega-kernel loop and issue policy: scan, spin, buffer, retry, completion observation. | Task publication semantics, dependency semantics, spatial mapping, event semantics. |
| Scheduler | Logical readiness from explicit task dependency attrs. | Spatial resources, backend peer identity, CUDA streams, EventTensor wait conditions. |
| Dispatcher | Spatial mapping from task attrs and worker context to work cursors. | Dependency order, event semantics, backend primitive behavior. |
| EventTensor | Event tensor object schema, wait-count semantics, notify/wait/trigger attrs, sync-task completion conditions, lowering hook names. | Runtime loop policy, spatial mapping, scheduler task dependency semantics, CUDA atomics, NVSHMEM rank/team behavior. |
| Platform | CUDA execution facts and launch/error mechanics. | Backend team identity or task readiness. |
| Backend | NVSHMEM team/rank/symmetric-storage facts and communication primitives. | Scheduler policy or dispatcher placement. |
| Driver | Host-provided platform/backend execution resources. | Target args, task records, event records, operator argument packs. |

The runtime loop calls the other components. It is not a scheduler replacement
and must not invent dependencies or spatial placement. It owns the loop that
decides what to try, when to spin, when to defer, and when to observe
completion. The runtime execution model owns task publication, sealing, and
host/device orch placement.

## Device-Side Interaction

The runtime loop is intentionally replaceable. A runtime may scan statically,
spin on a ready task until spatial resources are available, or keep a small
ready-but-not-issued buffer. The common interaction is:

```text
runtime loop:
  for tasks according to its loop discipline:
    ask scheduler whether explicit task deps are ready
    for sync-only tasks, ask EventTensor whether wait conditions complete
    ask dispatcher whether this worker has a work cursor for this task
    invoke operator body or sync-task event handler
    report event progress and task progress
```

The scheduler does not have to provide one fixed `next_ready()` API. It may
provide readiness queries, completion observation, and strategy-specific state.
The dispatcher may branch on builtin task attrs and dispatcher-owned attrs, but
it must not infer task kind from `op == null`, pointer arguments, tensor names,
or event attrs.

## Runtime Candidates

The current PR-required runtime execution models are:

- `host-orch`: host orch builds and seals a complete task arena before launch.
- `seeded-orch`: host seeds launch facts and device orch publishes concrete
  task records inside the mega-kernel.

The current accepted runtime loop candidates are:

- `block_tile_runtime`: tiled CUDA mega-kernel loop for distributed tile-grid
  tasks. In the current PR, the active distributed example targets are GEMM-RS
  and AG-GEMM.
- `grid_stride_runtime`: fixed-size independent range loop for vector-style
  stages and tiny decode work.
- `single_work_item_runtime`: minimal one-work-item loop for contract tests and
  small sync targets.

Rejected or future runtime candidates:

- problem-specific GEMM/communication runtime in Megacu core;
- host-dispatched per-operator CUDA kernels as the Megacu path;
- stream-based internal scheduling, because the host stream only enqueues and
  synchronizes the outer mega-kernel.

## Architecture Comparison

| System | Useful lesson | Boundary Megacu should not copy |
| --- | --- | --- |
| MegaKittens/Hazy | Explicit instruction/barrier readiness and end-to-end decode pipeline shape. | Fixed instruction ABI, fixed worker roles, model-specific tensor slots. |
| Simpler/PTO Runtime | Clear orchestrator/scheduler/worker split and task handle flow. | TensorMap dependency inference and host queue execution as Megacu semantics. |
| Triton-distributed | Torch/NVSHMEM launch facts, symmetric storage, rank-aware tests, device communication protocols. | Embedding all scheduling and communication protocol choices inside one hand-written kernel. |

Megacu sits between Simpler's host DAG runtime and specialized mega-kernel
systems:

- Like MegaKittens, Megacu uses explicit readiness and a mega-kernel-shaped
  execution path.
- Unlike MegaKittens, Megacu does not require a fixed instruction record,
  fixed worker roles, or model-specific tensor slots.
- Like Simpler, Megacu has an orchestrate layer that records task objects.
- Unlike Simpler, Megacu does not infer dependencies from tensor access tags
  and does not dispatch through a host scheduler thread.
- Like Triton-distributed, Megacu treats distributed launch, rank facts,
  symmetric storage, and device-side communication as first-class facts.
- Unlike Triton-distributed, Megacu exposes reusable task/event, dispatcher,
  scheduler, runtime, platform, and backend components instead of embedding all
  protocol choices inside one handwritten kernel.

## Hard Rules

- Dependencies are explicit attrs only.
- Event Tensor is an orchestration object, not hidden runtime state.
- Operators never manually call EventTensor notify/wait/trigger, backend
  signal/wait, or scheduler APIs.
- Missing dependency/event attrs are user responsibility; Megacu does not infer
  or repair them from raw arguments.
- Launch adapters normalize process/rank/backend setup into the driver; they
  do not choose dispatcher placement or scheduler order.
- The runtime owns execution model and internal loop; scheduler and dispatcher
  remain narrow services.
- Megacu examples must be composed from submitted tile/range operator tasks.
  Handwritten large fused/persistent kernels belong only in baseline variants,
  not in the Megacu path.

## Required PR Examples

This PR's active example scope is:

- GEMM-RS: GEMM tile producer tasks feed sync-only EventTensor readiness tasks,
  which then feed reduce-scatter tile/region consumer tasks.
- AG-GEMM: all-gather segment or tile producer tasks feed sync-only
  EventTensor readiness tasks, which then feed GEMM tile consumer tasks.
- Tiny decode pipeline: local tile/range operator stages prove the same runtime
  model outside distributed GEMM communication.

All three Megacu variants should use the same architectural pattern:

```text
orchestrate(driver, raw args...)
  -> submit operator tasks and sync-only tasks
  -> attach explicit scheduler and EventTensor attrs
  -> runtime execution model publishes/seals task records
  -> runtime loop asks scheduler, dispatcher, and EventTensor what can run
  -> operator table invokes tile/range operators by slot
```
