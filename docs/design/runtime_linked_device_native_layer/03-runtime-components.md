# Runtime Components

Dispatcher, scheduler, platform, backend, and target runtime are linked C++
components that run inside the orchestrate call. They are not compiler passes,
and normal programmers should not call them directly.

## PR #4 Proof Boundary

PR #4 should make these components tiny but mighty:

- general in ownership and interfaces;
- minimal and naive in implementation;
- phased only;
- enough for 1-host/1-device and 1-host/2-device GEMM+AllReduce;
- no MPI, torch-distributed, or overlap support.

Problem-specific shortcuts may live in the example only. Shared component APIs
must not bake in GEMM+AllReduce names, CUDA launch arguments, NVSHMEM team
arguments, or fixed participant enums.

## Target Runtime

Target runtime owns the hidden execution frame:

```cpp
using gemm_ar_arguments = std::tuple<
    const float *, const float *, float *, float *, int, int, int>;

struct call_frame {
  cuda_nvshmem::driver &driver;
  gemm_ar_arguments arguments;
  target_capability capability;
  component_set components;
};
```

The user sees the host-call signature, for example
`gemm_allreduce_phased_orchestrate(driver, a, b, partial, out, m, n, k)`. The
target runtime sees the full call frame and sequences dispatch, scheduling,
platform/backend access, and status propagation. PR #4 intentionally does not
add a heavy platform/backend validation layer.

Contract:

- consume the target's direct C++ arguments without wrapping them in a generic
  argument schema or payload object;
- convert component failures into `megacu::status`;
- keep platform/backend execution resources in the driver rather than target
  arguments;
- keep target arguments, argument storage, linked component sets, capability
  envelopes, and target-local scratch/events out of the driver;
- never materialize static metadata sections.

## Attribute Interface

Megacu core owns only a typed attribute container:

```cpp
struct attr_set {
  template <class Attr>
  bool has() const;

  template <class Attr>
  const Attr *get() const;
};
```

Components own the attributes they understand:

```cpp
namespace megacu::dispatcher {
struct tile_grid { int m; int n; int k; };
struct rank_policy_local_then_peer {};
}

namespace megacu::scheduler {
struct depends_on { task_ref predecessor; };
struct event_tensor_shape { int first; int second; };
struct phase_compute_then_comm {};
}

namespace megacu::backend {
struct requires_symmetric_storage { int arg_index; };
}
```

Contract:

- components read only their own attributes or shared attributes explicitly
  documented as cross-component contracts;
- required unknown attributes return `unsupported`;
- optional unknown attributes are ignored by components that do not own them;
- diagnostics may use names, but execution semantics must not depend on string
  lookup.

This replaces the previous fixed participant-attribute enum list. Virtual
participants remain a possible future pattern, but they must be implemented on
top of the composable attribute mechanism.

## Dispatcher

The dispatcher owns spatial mapping. For the general design, it maps submitted
tasks plus component attributes to compact work assignment:

```cpp
struct dispatch_request {
  driver_ref driver;
  task_list tasks;
};

struct dispatch_state;

megacu::status map(dispatch_state &out, const dispatch_request &request);
```

Contract:

- derive work from runtime problem values and task attributes;
- retarget the same task submission for 1-host/1-device and 1-host/2-device;
- map local work and peer work without changing the author-facing code;
- expose compact cursors to scheduler/operator code;
- avoid heap-heavy tables in the hot path.

PR #4 dispatcher:

- may only understand one tiny phased GEMM+AllReduce task pattern;
- must still live behind the dispatcher interface;
- maps single-device work locally;
- maps two-device work to local plus peer reduction work using backend
  resources from the driver;
- does not support overlap co-residency.

## Scheduler

The scheduler owns temporal progress and dependency consumption:

```cpp
struct schedule_request {
  driver_ref driver;
  task_list tasks;
  dispatch_state dispatch;
};

megacu::status run(const schedule_request &request);
```

Contract:

- consume only explicit dependency attributes such as
  `scheduler::depends_on(task_ref)`;
- choose when linked operators run;
- reject unsupported dependency or progress requirements before launch when
  possible;
- never recompute rank/peer placement owned by the dispatcher;
- never infer dependencies from tensor access lookup, tensor names, event
  names, input/output/inout sets, or pointer aliasing;
- never perform fallback dependency checking to infer or repair omitted
  dependency attributes.

PR #4 scheduler:

- ASAP only;
- run any task as soon as all explicit dependencies are complete;
- use submission order only as a deterministic tie-breaker among ready tasks;
- one explicit dependency path from GEMM to sync-only readiness to phased
  all-reduce;
- no overlap progress guard;
- no dynamic queues.

## Dependency Model

Megacu has two task kinds and one explicit event object:

- **operator tasks**: execute a linked native operator body with raw arguments;
- **sync-only tasks**: execute no operator body and exist only to reshape,
  join, split, or trigger readiness between other tasks.
- **event tensors**: orchestration-owned readiness objects declared in the
  orchestrate scope and referenced by task attributes.

Both task kinds return `task_ref` and can be used with explicit dependency
attributes. Sync-only tasks are Megacu's native way to carry the Event Tensor
middle node, but the event tensor itself is not hidden inside a runtime-owned
stage kernel:

```text
event tensor declaration
producer operator task --events::publish--> sync-only task --events::join-->
consumer operator task --events::acquire-->
```

The sync-only task and event handlers are declarative. Users do not program
waits, notifies, semaphores, NVSHMEM polls, CUDA stream waits, or scheduler
calls. Platform/backend components lower event tensor attributes into the
concrete counters, waits, signals, or ready-queue pushes needed by the selected
scheduler policy.

Tasks pass raw operator arguments and explicit attributes:

```cpp
auto ready_events = phase.event_tensor(attrs(
    events::shape{m_tiles, n_tiles},
    cuda_nvshmem::events::symmetric_i32_storage{events},
    cuda_nvshmem::events::team_scope{}));

auto gemm = submit(
    gemm_op,
    driver, a, b, partial, problem,
    attrs(
        dispatcher::tile_grid{m_tiles, n_tiles},
        events::publish(ready_events)));

auto ready = sync(attrs(
    scheduler::depends_on(gemm),
    events::join(ready_events)));

submit(
    ar_op,
    driver, partial, out, problem,
    attrs(
        scheduler::depends_on(ready),
        events::acquire(ready_events)));
```

Dependency sources:

- explicit `scheduler::depends_on(task_ref)` attributes;
- explicit sync-only tasks that represent readiness transforms;
- explicit event tensor handler attributes such as `events::publish`,
  `events::join`, and `events::acquire`.

Tasks do not declare `input`, `output`, or `inout` access. Pointer/scalar
arguments are passed through to linked operators as raw values. Components may
consume facts they explicitly own, such as backend team facts or dispatcher
tiling attributes, but the scheduler must not create
hidden read-after-write or write-after-write edges by looking up tensor
operations or pointer aliases. The dependency list is runtime execution state
for one call. It is not program IR and is not converted into static schedule
sections.

Missing dependency or event tensor attributes are user errors by contract, but
Megacu does not try to prove that they are missing from raw arguments. The thin
runtime consumes only the explicit dependency and event attributes it is given.

## Platform

Platform components own accelerator execution resources. CUDA can be the first
platform, but CUDA views should be platform-owned resources in the driver, not
fixed Megacu core fields or target arguments.

PR #4 CUDA platform responsibilities:

- expose simple CUDA launch facts, such as whether a stream is present;
- keep CUDA error conversion at the native operator boundary;
- avoid deep stream/device/capability validation in this PR.

It does not own backend PE identity, task dependencies, or participant mapping.

## Backend

Backend components own communication resources and primitives. NVSHMEM can be
the first backend, but NVSHMEM team resources should be backend-owned resources
in the driver, not fixed Megacu core fields or target arguments.

PR #4 NVSHMEM backend responsibilities:

- expose simple team facts, such as team size and whether remote PEs exist;
- lower CUDA+NVSHMEM event tensor handlers to the device-side fences, waits,
  signals, and remote loads needed inside the mega-kernel;
- avoid deep team/symmetric-storage validation in this PR.

It does not choose scheduler ordering or expose MPI/torch-distributed adapter
contracts in PR #4.

## Operators

Operators have direct native signatures. The component layer records enough
implementation-private call state to invoke linked native symbols, but it must
not expose a public `op_args`, `input`, `output`, or `inout` abstraction.
Operators should not implement dependency readiness manually; Megacu wraps
operator execution with internal readiness handling derived from dependency,
sync-only, and event tensor attributes.

```cpp

```cpp
using native_arg_pack = implementation_private;

struct operator_call {
  operator_id id;
  native_arg_pack args;
  attr_set attrs;
};
```

Contract:

- each operator is typed by its linked C++/CUDA signature;
- submitted raw arguments must match that signature;
- `submit` consumes the number of raw arguments declared by the operator
  signature, followed by an optional trailing `attr_set`;
- native symbols remain handwritten C++/CUDA/NVSHMEM code;
- operator signatures are not one global fixed Megacu ABI;
- public task APIs do not classify arguments as reads, writes, or inouts.
- operator bodies do not program scheduler waits, event notifies, or
  dependency-management polls directly.

## Sync-Only Tasks

Sync-only tasks are first-class tasks with no operator body. They are used when
readiness is the work:

- join many producer tiles before a consumer region;
- split one producer readiness into many consumer regions;
- reshape producer task coordinates into consumer task coordinates;
- express data-dependent event updates or task triggering;
- represent barrier-like readiness without a dummy compute operator.

This mirrors the Event Tensor idea that an event can be a first-class node
between producer and consumer tasks. Megacu keeps both the event tensor and the
sync-only node explicit, then hides the implementation. A sync-only task lowers
through event tensor handlers; it is not justified by CUDA stream order. In a
static scheduler, that can mean inline counter updates and waits around
operator bodies. In a dynamic scheduler, sync completion can push ready
consumer tasks into a device queue. For CUDA+NVSHMEM, the backend owns whether
those counters are local global memory, symmetric memory, remote loads, or
backend-specific wait primitives.

Rejected PR #4 design: runtime-owned event stage kernels. They hide Event
Tensor semantics too far below the orchestrate surface and encourage producer
or consumer operators to carry readiness code. Also rejected for the Megacu
path: putting a handwritten monolithic fused mega-kernel in the example and
calling that Megacu composition. That belongs only as a baseline. PR #4 instead
uses a native Megacu launcher that links common device-entry running logic,
explicit scheduler, dispatcher, platform, backend event tensor, and an operator
table. The CUDA platform launches one generic mega-kernel; the scheduler
chooses the next logical task; the dispatcher maps work to CTA/tile placement;
the backend lowers event tensor attrs; and the operator table invokes the
selected operator body.

## No Static Sections

Runtime components must not return static section objects such as
`dispatch_section` or `schedule_section`. Tests should inspect runtime behavior:

- task dependencies are consumed only from explicit dependency attributes;
- sync-only tasks can sit between operator tasks without invoking an operator
  body;
- dispatcher maps phased work for 1-host/1-device and 1-host/2-device;
- scheduler validates the expected ASAP dependency order;
- platform/backend remain thin accessors rather than heavy validators.
