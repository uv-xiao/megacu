# Program Model

The authored unit in Megacu is now an orchestrate program written in C++.

That program is the logical orchestration description that CMake will compile
and link into a concrete target. It is not a plugin artifact, and it is not a
generic runtime-loaded module.

## Terms

- **Orchestrate program**: the C++ function users write to declare tasks,
  events, resources, and `run`. In the first implementation this should be a
  small C++ program descriptor plus a direct runtime wrapper, both compiled
  into the orchestrate target.
- **Operator** or **named op**: a user kernel entry identified by a C++ op tag
  and implemented by a native CUDA/C++ kernel or callable device function.
- **Task** or **submission**: one invocation of an operator over a logical
  workset, with typed arguments and event dependencies.
- **Domain**: a logical index space such as tiles, tokens, experts, or ranks.
  A domain says what work exists. It does not say which CUDA block, rank, CTA,
  or scheduler lane will execute that work.
- **Domain point**: one element of a domain, such as tile 17. Domain points may
  be mapped to backend-specific execution coordinates by the dispatcher.
- **Workspace**: typed temporary or persistent storage supplied to the
  orchestrate program. Megacu does not own allocation policy; a workspace view
  describes buffers the program and kernels may use.
- **Event**: a typed logical dependency between tasks. The author gives an
  event a name for diagnostics, but lowering assigns the event slot and backend
  representation.
- **Virtual participant**: a logical endpoint such as producer, consumer,
  source rank, destination rank, or pipeline lane. It is virtual because the
  orchestrate program uses it before the dispatcher maps it to backend rank,
  peer, CTA, or lane ids.
- **Kernel context**: the lowered device-side context passed to operators so a
  kernel can resolve its current domain point, virtual peers, event endpoints,
  workspace slices, and backend primitives without string lookup.

Names are labels. They are useful for diagnostics and generated metadata, but
they are not runtime lookup keys. The implementation must use typed tags and
lowered metadata for event, domain, operator, and virtual-participant
resolution.

## What The Orchestrate Program Must Express

The orchestrate program should express only the information needed before
build-graph lowering:

- named ops
- resources
- explicit events
- logical work domains
- virtual participants when communication crosses logical endpoints
- tasks/submissions
- orchestration order and control flow
- optional mapping hints for the dispatcher

That is the whole user-visible model.

The first public headers should be:

- `include/megacu/program.h`: `program_builder`, domain/event/participant/
  resource/submit builders, and extent tags.
- `include/megacu/executor.h`: compiled target executor, runtime binding, and
  `run`.
- `include/megacu/views.h`: typed resource views.
- `include/megacu/backends/nvshmem.h`: first backend runtime and device views.

## First Implementation Shape

To make build-time lowering concrete, the first implementation should not try
to inspect arbitrary C++ function bodies. Users write a C++ program descriptor
with an explicit `describe(...)` method, then expose an ordinary runtime
function that calls the compiled executor.

```cpp
struct event_copy_program {
  static void describe(megacu::program_builder &p) {
    auto tiles = p.extent<tiles_extent>("tiles");
    auto tile = p.domain<tile_domain>("tile", tiles);
    auto producer = p.participant<producer_lane>("producer");
    auto consumer = p.participant<consumer_lane>("consumer");
    auto workspace = p.resource<workspace_slot, event_copy_workspace>("workspace");
    auto events = p.resource<event_storage_slot, megacu::event_storage_view>("events");

    auto ready = p.event<ready_event>(
        "ready",
        tile,
        megacu::remote_event{.from = producer, .to = consumer, .storage = events});

    p.submit(
        ops::write_then_signal{},
        megacu::over(tile),
        megacu::place(producer),
        megacu::args().payload(workspace.payload).signal(ready.release()));

    p.submit(
        ops::wait_then_check{},
        megacu::over(tile),
        megacu::place(consumer),
        megacu::args().wait(ready.acquire()).payload(workspace.payload));
  }
};

void cuda_nvshmem_event_copy_orchestrate(
    event_copy_workspace workspace,
    megacu::event_storage_view events,
    megacu::nvshmem_team_view team,
    std::int32_t tiles) {
  megacu::executor<event_copy_program> exec{team};
  exec.bind<workspace_slot>(workspace);
  exec.bind<event_storage_slot>(events);
  exec.run(megacu::extent<tiles_extent>(tiles));
}
```

`describe(...)` is the concrete program surface that CMake/native build tooling
can compile into metadata. The runtime wrapper is the direct call surface used
by applications and framework wrappers. This keeps build work out of runtime
C++ while avoiding a hidden parser for arbitrary C++.

## Named Ops

The public operation concept is a named op.

A named op refers to a native kernel implementation known to the native build
system. That kernel may be:

- a regular CUDA/C++ kernel
- a backend-assisted kernel that calls send/wait/signal primitives inside its
  body
- a kernel later stitched into a persistent lowering form by build-graph logic

The orchestrate program does not distinguish these as different public op
kinds.

Initial API shape:

```cpp
namespace ops {
struct write_then_signal {
  static constexpr auto name = "write_then_signal";
};

struct wait_then_check {
  static constexpr auto name = "wait_then_check";
};
}
```

A named op is a type or symbol known to the native build graph. The user should
not construct an op descriptor manually.

The CMake target maps op tags to implementation symbols:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_event_copy
  PROGRAM event_copy_program
  SOURCES event_copy_orchestrate.cc
  KERNELS event_copy_kernels.cu
  OPS
    write_then_signal=write_then_signal_kernel
    wait_then_check=wait_then_check_kernel
  COMPONENTS cuda_nvshmem_static
)
```

The op keys in `OPS` are build-time names that must match the op tags'
`name`. They are not runtime lookup keys.

## Resources

Resources are typed handles representing input, output, inout, and
scratch/storage bindings. The program should not force users to think in raw
resource ids.

Example shapes:

- `workspace_view`
- `event_storage_view`
- `routing_replay_view`
- `k_cache_view`

Build-time lowering may map them to flat resource indices inside target
internals.

Initial API shape:

```cpp
namespace megacu {
struct workspace_view;
struct event_storage_view;
struct tensor_view;
}
```

Resource views are explicit C++ arguments to the orchestrate function. There is
no generic resource map, no string lookup, and no public resource id plumbing in
normal examples.

`workspace_view` is not magic global memory. It is a typed view over caller-
provided storage:

```cpp
struct event_copy_workspace {
  megacu::span<std::byte> scratch;
  megacu::span<std::uint64_t> payload;
};
```

The orchestrate program receives this view, passes typed slices to submissions,
and lowering records which resource slots kernels need. Allocation remains with
the caller or framework wrapper.

## Events

Events remain first-class because they are the visible coordination mechanism
between named ops.

The orchestrate program should expose:

- event handles
- wait
- signal
- ordering/scope only where correctness requires them

Rich coordinate-heavy event spaces are allowed as builder helpers, but they
should not become the dominant public mental model. The dispatcher and lowering
can flatten them into compact target metadata.

Initial API shape:

```cpp
struct ready_event;

auto ready = orch.event<ready_event>(
    "ready",
    tile,
    megacu::remote_event{
        .from = producer,
        .to = consumer,
        .storage = events});

args()
    .signal(ready.release())
    .wait(ready.acquire());
```

Event handles own logical coordination only. Backend signal objects, event
storage offsets, memory-order details, and remote-rank mappings are target
internals owned by lowering and backend adapters.

The event name `"ready"` is diagnostic. The typed tag `ready_event` is the
stable program identity. Lowering turns `ready_event` into an event slot and an
event-storage layout. At runtime, the backend adapter resolves that slot plus
the current domain point and virtual participants into the concrete NVSHMEM
signal address or equivalent backend object.

## Logical Work Domain

The earlier design exposed too many ids, coordinates, and dimensions. The
program model should replace that with a simpler concept: logical work domain.

Examples:

- tokens
- tiles
- experts
- ranks
- blocks

The program can attach submissions to domains or domain slices without exposing
the final worker/rank mapping. The dispatcher consumes this information during
build-graph lowering.

This is the information the dispatcher needs from the program model:

- op handle
- resource usage
- event dependencies
- domain/workset membership
- optional placement hints

It does not need the user to pre-assign worker ids or scheduler lanes.

Initial API shape:

```cpp
struct tile_domain;

auto tile = orch.domain<tile_domain>("tile", tiles);
orch.submit(ops::write_then_signal, megacu::over(tile), args);
```

The domain name is for authoring and diagnostics. Lowering may replace it with
compact ids in generated metadata, but those ids are not public authoring API.

The first implementation only needs one-dimensional domains. Multi-dimensional
helpers can be added later only if examples need them.

## Virtual Participants

Communication often needs "the peer task" before the backend rank or lane is
known. Megacu models that through virtual participants.

Initial API shape:

```cpp
struct producer_lane;
struct consumer_lane;

auto producer = orch.participant<producer_lane>("producer");
auto consumer = orch.participant<consumer_lane>("consumer");

orch.map(producer, megacu::placement::rank(0));
orch.map(consumer, megacu::placement::rank(1));
```

For static examples, the mapping may be stated in the orchestrate program or in
the CMake target configuration. Either way, runtime kernels do not use these
names as strings. Lowering creates a placement table that maps virtual
participants and domain points to backend-native ids.

Virtual participants are allowed only when they remove raw rank/worker ids from
the public program. They must not become a second scheduler API.

## Tasks And Submissions

The public task concept is `submit`: one named op over one logical workset with
typed arguments and dependencies.

Initial API shape:

```cpp
orch.submit(
    ops::write_then_signal{},
    megacu::over(tile),
    megacu::place(producer),
    megacu::args()
        .workspace(workspace)
        .signal(ready.release()));
```

`submit` records:

- named op
- logical workset
- optional virtual participant placement
- typed resource arguments
- event acquire/release dependencies
- optional placement hints when required by a target

`submit` must not expose:

- public `task_kind`
- public `task_desc`
- raw packed descriptor fields
- scheduler lane ids
- final rank/worker ids

The implementation may lower submissions into compact task records, but those
records are target internals.

## How Users Author Orchestrate Programs

The intended authoring flow is API collection, not trait filling and not raw
descriptor construction.

The concrete authoring surface should be ordinary C++ orchestration code that
is compiled by CMake. The `program` is a semantic model, not a promise that
runtime C++ must expose a build-capable API.

Shape:

```cpp
struct tiles_extent;
struct tile_domain;
struct producer_lane;
struct consumer_lane;
struct ready_event;
struct workspace_slot;
struct event_storage_slot;

struct event_copy_program {
  static void describe(megacu::program_builder &p);
};
```

The concrete structure is:

- author one program descriptor with `describe(...)`
- attach resources/events/domains through small builders
- let the build graph lower the descriptor into target metadata
- expose an ordinary runtime function that binds values and calls
  `executor<program>::run(...)`

For the first implementation, the required public surface is:

- `megacu::program_builder`
- `megacu::executor<Program>`
- `megacu::extent`
- `megacu::domain`
- `megacu::event`
- `megacu::participant`
- `megacu::over(domain)`
- `megacu::place(participant)`
- `megacu::args()`
- `megacu::program_builder::submit(...)`
- `megacu::executor::bind(...)`
- `megacu::executor::run(...)`

Anything else must justify why event, task/submission, schedule, or kernel
contracts cannot work without it.

## Fine-Grained Overlap

The program model supports fine-grained compute/communication overlap in two
ways:

1. explicit event relationships between named ops
2. backend primitives inside named kernel bodies

The second path is important. A named kernel can perform fine-grained
communication or signaling in the middle of its device code. Megacu does not
need a public "fragment op" abstraction to allow that.

If build-graph lowering can safely stitch several named ops into a single
persistent execution form, that is a lowering decision, not a public program
concept.

Device-side backend primitives must use a lowered kernel context, not authoring
strings:

```cpp
extern "C" __global__
void write_then_signal_kernel(
    megacu::cuda::kernel_context ctx,
    event_copy_workspace workspace) {
  auto tile = ctx.domain_point<tile_domain>();
  auto peer = ctx.peer<consumer_lane>(tile);
  auto ready = ctx.event<ready_event>(tile, peer);

  workspace.payload[tile.linear] = make_payload(tile.linear);
  megacu::nvshmem::signal(ctx, ready, 1);
}
```

`ctx.peer<consumer_lane>(tile)` is where the virtual participant is resolved to
the backend-native peer for this lowered target. `ctx.event<ready_event>(...)`
is where the event tag is resolved to the backend-native event endpoint for
this domain point and peer. There is no device-side string lookup.

## What The Program Does Not Contain

The orchestrate program should not contain:

- scheduler implementation choice
- kernel-shape choice
- backend/platform selection
- raw packed descriptors
- final worker/rank/resource indices

Those belong to build-graph lowering and to target internals.

## Program Implementation Target

The first code slice should prove the program model through
`examples/cuda_nvshmem_event_copy/` and compile checks under `tests/build/`.

Program-model evidence:

- a compile-only example using the public builder APIs above
- no user-authored packed descriptors
- no public task-trait field filling
- no string resource lookup
- no device-side event-name lookup
- generated metadata showing domain, participant, and event tags resolved to
  compact slots
- direct call of the compiled orchestrate function from runtime C++
