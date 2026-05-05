# Programming Surface

Megacu user code is ordinary C++ orchestration linked into a target. The user
should not author an IR, call a materializer, or write dispatcher/scheduler
boilerplate.

The author-facing surface has three ideas:

1. a host-called orchestrate entry with a driver plus target-specific
   CUDA-kernel-like arguments;
2. small task/operator submission helpers that pass raw arguments to linked
   operators, plus sync-only task helpers that declare readiness transforms;
3. linked components that consume the submitted task graph and driver resources
   behind the authoring surface.

This follows the useful split seen in Simpler/PTO Runtime:
`aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args)` reads tensors
and scalars from one argument object, builds `Arg` objects with inputs,
outputs, and inouts, and submits kernel ids through small helpers. Megacu should
not copy PTO internals. In particular, Megacu should not infer dependencies by
looking up tensor operations the way simpler's tensormap/ringbuffer runtime can.
Megacu should also avoid adopting `input/output/inout` as a task argument
abstraction. For CUDA-like operators, pointers, scalars, and small descriptors
can be passed raw; dependencies and other component facts are the explicit
attributes.

## PR #4 Proof Boundary

PR #4 needs a tiny phased proof with 1-host/1-device and 1-host/2-device
coverage. The proof may use minimal and naive dispatcher/scheduler components,
but they should still be components. It should not expose CUDA launch handles
or NVSHMEM team handles as target arguments. Those execution resources belong
to the driver.

The retired communication-progress variant is out of PR #4. For this PR,
kernels that include communication should
be represented as one fused phased operator rather than split into several
standalone operators that force dispatcher/scheduler complexity. Whether the
general design needs finer communication decomposition remains future work.

## Orchestrate Entry

Megacu orchestrate is called from host code. For CUDA+NVSHMEM, that host code
already owns asynchronous execution concepts such as CUDA streams, device
selection, launch/synchronize policy, and distributed runtime state. The
orchestrate entry should therefore look like a host function that enqueues one
target megakernel through a driver, not like a generic runtime graph call. That
one Megacu megakernel runs the common device-side dispatcher/scheduler loop and
invokes linked operator kernels internally; host code still sees one
orchestrate call.

The public ABI should not be a fixed list such as `workspace, events, launch,
team, problem`, and it should also not force all target values through an
over-wrapped `tensor()/scalar()/resource()` frame. A target exports one
target-specific entry whose arguments look like CUDA kernel arguments:
device pointers, scalar values, and small POD descriptors when needed.

```cpp
extern "C" megacu::status gemm_allreduce_phased_orchestrate(
    megacu::cuda_nvshmem::driver &driver,
    const float *a,
    const float *b,
    float *partial,
    float *out,
    int m,
    int n,
    int k);
```

The driver is the execution abstraction. It is not a workload tensor/scalar
bag. For CUDA+NVSHMEM it owns or references:

```text
CUDA platform resources:
  stream used to enqueue the target megakernel
  device/context identity
  launch capability checks and CUDA error conversion

NVSHMEM backend resources:
  PE/team identity
  symmetric storage/session validation
  device-side primitive availability

Megacu target resources:
  none; linked components, target scratch, events, and argument storage belong
  to the linked target or target arguments, not to the driver
```

Driver inclusion rule:

- Include resources owned by the host execution environment that are required to
  launch, synchronize, or describe platform/backend execution: CUDA stream,
  device/context identity, NVSHMEM PE/team identity, team size, and distributed
  process facts.
- Exclude target arguments, target argument storage, operator argument packs,
  target scratch/events, linked component sets, and capability envelopes. Those
  are workload or linked-target state, and putting them in the driver would
  recreate a generic runtime bag.

The target declares its argument contract as the direct C++ signature. There is
no separate argument schema, payload object, or generic frame:

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    megacu::cuda_nvshmem::driver_view driver,
    float const *a,
    float const *b,
    float *partial,
    float *out,
    void *events,
    gemm_ar_problem problem);
```

Any required pointer/storage/layout/shape preconditions belong to that target's
own contract and tests. Megacu does not infer missing dependency edges or add a
fallback argument checker around raw target arguments.

## Driver Resources

Megacu core should not define public `cuda::launch_view` or
`nvshmem::team_view` fields in a shared argument object. Platform and backend
components may define their own resource views, but those views are owned by
the driver and read by components that understand them:

```cpp
namespace megacu::cuda_nvshmem {
struct driver {
  template <class Resource>
  Resource get(resource_key key) const;

  cudaStream_t stream() const;
  int device() const;
};
}

namespace megacu::cuda {
struct stream_resource;
struct device_resource;
}

namespace megacu::nvshmem {
struct team_resource;
struct symmetric_heap_resource;
}
```

The programmer normally does not pull these resources out. The linked platform
and backend components validate and use them. Example-local validation may read
them in PR #4 only when necessary to produce clear status errors.

For distributed execution, each host process or framework rank calls the same
orchestrate entry with its local driver. The driver is where the process model
is normalized: direct local process, `nvshmrun`, MPI-launched setup, or a future
torch-distributed adapter can all produce the CUDA+NVSHMEM resources the linked
components require. PR #4 only needs the direct local 1-device and 2-device
single-host paths, but the abstraction boundary should already be clear.

## Authoring Code

The programmer writes orchestration logic, not component calls:

```cpp
extern "C" megacu::status gemm_allreduce_phased_orchestrate(
    megacu::cuda_nvshmem::driver &driver,
    const float *a,
    const float *b,
    float *partial,
    float *out,
    int m,
    int n,
    int k) {
  megacu::scope phase(driver);

  auto gemm = phase.submit(gemm_tile_kernel,
      a, b, partial,
      attrs(tile_grid{m, n, k}));

  auto ready = phase.sync(
      attrs(scheduler::depends_on(gemm)));

  phase.submit(phased_allreduce_kernel,
      partial, out,
      attrs(scheduler::depends_on(ready)));

  return phase.finish();
}
```

This is author-facing pseudocode. `phase.submit(...)` forwards raw operator
arguments and explicit attributes to the linked components. `phase.sync(...)`
declares a sync-only task: a readiness node with no operator body. The
execution reality may call target validation, dispatcher, scheduler, platform,
backend, and native operators, but programmers should not spell those calls
manually in normal orchestration code.

`phase.submit` uses the linked operator signature as the thin parsing rule. For
an operator with N native arguments, the first N values after the operator
reference are raw operator arguments. An optional trailing `attr_set` attaches
component attributes. No argument builder is needed.

## Component-Provided Attributes

Megacu core should not hard-code a closed `participant_attrs` enum set. It
should provide a typed attribute composition interface. Components provide the
attribute types they understand; programmers compose them at submit sites or
scopes:

```cpp
auto hints = attrs(
    dispatcher::tile_grid{m, n, k},
    dispatcher::rank_policy::local_then_peer{},
    backend::requires_symmetric(partial));
```

Rules:

- Megacu core owns the attribute container and typed lookup mechanism.
- Dispatcher, scheduler, platform, backend, and target components own their
  own attribute vocabularies.
- Unknown attributes are ignored only by components that do not own them.
  Required unknown attributes must produce a validation error before launch.
- Attribute names are diagnostics; semantics come from C++ types or stable
  component ids.
- PR #4 should use the smallest attribute set that proves phased
  1-host/1-device and 1-host/2-device execution.

Virtual participants can still exist as a future mechanism, but they should be
expressed through this extensible attribute interface rather than fixed Megacu
enums.

## Task Dependencies

Megacu handles dependencies as explicit task attributes. A submitted task does
not declare `input`, `output`, or `inout` access. It passes operator arguments
raw, then attaches typed attributes such as `scheduler::depends_on(gemm)`.

When readiness itself needs to be represented, the user declares an
orchestration-owned event tensor and attaches event handlers through task attrs
rather than writing waits or dummy kernels:

```cpp
auto ready_events = phase.event_tensor(attrs(
    events::shape(tiles_m, tiles_n),
    cuda_nvshmem::events::symmetric_i32_storage(events),
    cuda_nvshmem::events::team_scope()));

auto produced = phase.submit(
    producer,
    raw_args...,
    attrs(events::publish(ready_events)));

auto ready = phase.sync(attrs(
    scheduler::depends_on(produced),
    events::join(ready_events)));

phase.submit(
    consumer,
    raw_args...,
    attrs(
        scheduler::depends_on(ready),
        events::acquire(ready_events)));
```

This gives Megacu the Event Tensor middle node in native form:

```text
event tensor declaration
producer task --publish--> sync-only task --join--> consumer task --acquire-->
```

The sync-only task can model a join, split, coordinate reshape, or
data-dependent trigger. The event tensor object is visible to orchestration,
but platform/backend components own the concrete counters, waits,
notifications, remote loads, and queue pushes. Operator authors do not program
those mechanisms directly.

The scheduler receives a compact ready list or dependency list produced only
from dependency attributes. It runs tasks as soon as their explicit
dependencies are complete, using submission order only as a deterministic
tie-breaker. It should not infer dependencies by scanning tensor operations,
tensor names, event names, read/write sets, or pointer aliases, and the
programmer should not hand-build schedule sections.

Megacu does not perform fallback dependency checking. If the programmer omits a
required dependency attribute, Megacu does not infer or repair that edge from
the raw arguments. The target author is responsible for declaring the needed
ordering attributes.

For PR #4 phased GEMM+AllReduce:

```text
GEMM task
  raw args: A, B, partial
  attributes: dispatcher::tile_grid{m, n, k}

optional sync-only task
  raw args: none
  attributes: scheduler::depends_on(GEMM)

AllReduce task
  raw args: partial, OUT
  attributes: scheduler::depends_on(sync task or GEMM)
```

For 1-host/2-device, the backend component maps the all-reduce operator to
device-side NVSHMEM resources from the driver. The explicit dependency
attribute remains the same as the 1-device case.

## Operator Signatures

Operators should not share one fixed Megacu signature, and they should not
require a Megacu `op_args` builder. Each operator is a linked native symbol with
a direct C++/CUDA signature. The orchestration call passes raw values that match
that signature:

```cpp
using gemm_tile_sig = void(const float *a, const float *b, float *partial);
constexpr auto gemm_tile_kernel =
    megacu::operator_ref<"gemm_tile", gemm_tile_sig>();

using phased_allreduce_sig = void(float *partial, float *out);
constexpr auto phased_allreduce_kernel =
    megacu::operator_ref<"phased_allreduce", phased_allreduce_sig>();
```

The native CUDA/NVSHMEM implementation keeps its handwritten C++/CUDA
signature. The Megacu operator boundary uses the linked signature and component
attributes. Missing dependency attributes remain the target author's
responsibility; Megacu does not add a fallback checker that inspects raw
pointers to infer omitted edges.

## What Replaces Program IR

Instead of program records and materialization, the target owns:

- a configurable direct target argument signature;
- ordinary C++ orchestration code;
- submitted task/operator calls with raw arguments, explicit dependency
  attributes, and other component attributes;
- linked component implementations;
- linked native operator symbols.

The submitted task stream is runtime execution state for the current call. It
is not a stored compiler IR and it must not be materialized into static
dispatch/schedule/kernel sections.

## Public Surface Acceptance Criteria

- Workload ABI arguments are configurable by the target's direct signature.
- Target arguments are direct target-specific C++ arguments, shaped like CUDA
  kernel arguments rather than wrapped in a generic runtime bag.
- Platform/backend runtime resources are owned by the driver, not fixed target
  arguments.
- Normal programmer code does not call dispatcher or scheduler APIs directly.
- Attributes are component-provided and composable.
- Operator arguments are raw pointer/scalar/descriptor values matching the
  linked native operator signature.
- Task dependencies are explicit dependency attributes. They are not inferred
  from tensor access lookup, read/write declarations, or pointer aliasing.
- PR #4 proves phased 1-host/1-device and 1-host/2-device paths without MPI,
  torch-distributed, or retired communication-progress scope.
