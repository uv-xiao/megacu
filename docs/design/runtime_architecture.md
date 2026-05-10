# Runtime Architecture

Megacu is a thin runtime-linked layer for composing small operator kernels into
one host-launched mega-kernel. The user calls a target orchestrate function from
host code with raw CUDA-like arguments: pointers, scalars, descriptors, a CUDA
stream for launching the outer mega-kernel, and backend launch facts through the
driver. Megacu does not wrap tensor arguments in an input/output/inout argument
system and does not infer dependencies from pointer use.

## Orchestrator Model

An orchestrate call creates one call-owned orchestration frame:

```text
orchestration frame
  raw target args
  operator slots
  task records
  EventTensor records
  explicit attrs
  scheduler dependency groups
  linked runtime/component config
```

The frame can be built by either runtime execution model implemented in this
PR:

- `host-orch`: the host builds and seals a complete arena region before launch.
- `seeded-orch`: the host seeds launch facts, then a device control path builds
  and seals records inside the mega-kernel before worker blocks execute them.

Both execution models use the same recipe functions. The examples do not keep a
separate algorithm for host-built and device-built plans.

## Runtime Ownership

`runtime::device_persistent<ExecutionModel, Loop, Scheduler, Dispatcher,
EventTensor, Operators>` is the device-side mega-kernel runtime. It owns both
the execution model and the runtime loop.

```text
host orchestrate function
  -> prepare driver and raw target args
  -> build or seed a task arena
  -> launch one CUDA mega-kernel

device runtime
  -> bind/construct task arena through the execution model
  -> run the linked loop
  -> ask scheduler which explicit-dependency tasks are ready
  -> ask dispatcher whether this block has work for that task
  -> ask EventTensor whether sync-only wait conditions are complete
  -> invoke operator slot or complete sync task
```

The runtime loop is replaceable. The current examples use
`runtime::loop::block_tile`, which scans explicit-ready tasks and issues work to
CUDA blocks. Future loops can buffer ready tasks, spin on spatial resources, or
use a different issue policy without changing the scheduler API into a
user-programmed `run()` call.

## Component Boundaries

| Component | Owns | Does not own |
| --- | --- | --- |
| Orchestrate function | Raw target args, task submissions, EventTensor declarations, attrs, scheduler dependency groups. | Scheduler loop, backend communication internals, fallback inference. |
| Runtime execution model | Where and when task/EventTensor records are built, published, and sealed. | Dependency semantics, spatial mapping, event semantics. |
| Runtime loop | Mega-kernel internal issue policy. | Dependency inference, dispatcher placement rules, backend primitive meaning. |
| Scheduler | Readiness from explicit `depends_on` attrs. | CUDA streams, EventTensor wait conditions, spatial resources. |
| Dispatcher | Work cursor mapping from task attrs and block context. | Dependency order, event semantics, backend communication. |
| EventTensor | Logical event attrs and sync-only task completion conditions. | Scheduler task dependencies, runtime loop policy, spatial placement. |
| Platform | CUDA launch and device execution facts. | Backend rank/team identity. |
| Backend | NVSHMEM rank/team, symmetric storage, and remote communication facts. | Scheduler and dispatcher policy. |
| Driver | Host-provided platform/backend resources. | Target algorithm args and task records. |

## Hard Rules

- Dependencies are explicit attrs only. Large fan-in uses orch-owned dependency
  groups referenced by `scheduler::depends_on_many(...)`; it is still explicit
  scheduler state, not tensor-operation inference.
- EventTensor aggregate waits use EventTensor attrs such as
  `event_tensor::wait_strided(...)`; they define sync-task completion, not
  scheduler readiness.
- Missing dependency or EventTensor attrs are user responsibility.
- Megacu does not repair missing attrs, infer dependencies from raw pointers, or
  fall back to a safer host path.
- Operators do not call scheduler APIs or manually perform EventTensor waits or
  notifications.
- The CUDA stream only enqueues and synchronizes the outer mega-kernel.
- Handwritten large persistent kernels belong in baseline variants, not in the
  Megacu example path.
