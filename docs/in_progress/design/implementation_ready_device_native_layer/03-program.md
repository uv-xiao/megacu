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
- **Extent**: the runtime size of a domain. In the first example,
  `megacu::extent<tiles_extent>(tiles)` says how many tile domain points exist
  for this run.
- **Resource slot**: a typed program-declared input to the compiled target.
  `workspace_slot` and `event_storage_slot` are not allocation ids; they are
  typed binding points that `exec.bind<slot>(value)` fills at runtime.

Names are labels. They are useful for diagnostics and generated metadata, but
they are not runtime lookup keys. The implementation must use typed tags and
lowered metadata for event, domain, operator, and virtual-participant
resolution.

## Necessity Analysis

The design should keep only concepts that are required to run native kernels
with explicit coordination while preserving a thin runtime path.

| Concept | Why it is needed | If removed |
| --- | --- | --- |
| Operator | Connects an orchestrate submission to a native kernel implementation. | The scheduler would have no typed unit of executable work, or users would pass raw function/descriptors everywhere. |
| Task/submission | States that one operator runs over one workset with arguments and dependencies. | Event dependencies and resource usage would have to be encoded inside kernels or hidden descriptors, making scheduling opaque. |
| Domain | Names the logical workset before CUDA/rank mapping is chosen. | The API would expose CUDA block ids, ranks, lanes, or packed coordinates directly, tying the core to one backend strategy. |
| Extent | Supplies the runtime size of a domain without rebuilding the target. | Every new tile/token count would either require rebuilding or require a generic runtime graph builder. |
| Domain point | Gives kernels and metadata a way to talk about one logical work item. | Kernel code could not ask "which tile am I executing?" without backend-specific ids. |
| Resource slot | Gives the compiled target typed runtime inputs. | The runtime would need a generic `runtime_env`, positional argument convention, or string lookup. |
| Workspace | Groups caller-owned payload/scratch storage without making Megacu an allocator. | Users would either pass many unrelated buffers into every API or Megacu would need to own allocation policy. |
| Event storage | Separates synchronization storage from payload workspace. | Backend event state would be hidden inside payload buffers or allocated by Megacu, weakening explicit resource ownership. |
| Event | Represents a logical wait/signal dependency between tasks. | Communication ordering would live only in kernel code, so the scheduler/lowering could not inspect or validate it. |
| Virtual participant | Names logical communication endpoints before backend rank/lane resolution. | The public program would expose raw ranks/workers, or kernels would hard-code backend placement. |
| Kernel context | Carries lowered domain, participant, event, resource, and backend metadata to device code. | Kernels would need raw backend handles, string lookup, or handwritten per-target glue parameters. |

The removable parts are labels such as `"tile"`, `"ready"`, `"producer"`, and
`"workspace"`. They are not semantic requirements; they exist for diagnostics
and generated metadata. The required identities are the typed tags and lowered
slots.

This is why the first design keeps `domain`, `event`, `resource slot`,
`extent`, `participant`, and `kernel_context`, but rejects public raw
descriptors, string runtime environments, public scheduler classes, and raw
rank/worker ids in the normal authoring API.

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

For the first validation target:

- `tiles` is a runtime integer count: how many independent payload copies to
  run.
- `tile` is the logical domain over those copies. If `tiles == 128`, the
  domain has points `tile 0` through `tile 127`.
- a producer task and a consumer task are submitted for each tile point.
- `workspace` is caller-provided storage containing payload buffers and scratch
  used by those tasks.
- `events` is caller-provided event storage where the backend adapter can place
  the concrete signal/wait state for `ready_event`.
- `exec.bind<workspace_slot>(workspace)` binds the runtime workspace view to
  the resource slot declared by `p.resource<workspace_slot, ...>()`.
- `exec.bind<event_storage_slot>(events)` binds the runtime event-storage view
  to the event storage slot used by `p.event<ready_event>(...)`.
- `exec.run(megacu::extent<tiles_extent>(tiles))` supplies the runtime extent
  for the `tiles_extent` declared by `p.extent<tiles_extent>("tiles")` and then
  runs the lowered target.

```cpp
struct event_copy_program {
  static void describe(megacu::program_builder &p) {
    // Runtime value supplied by exec.run(megacu::extent<tiles_extent>(tiles)).
    auto tiles = p.extent<tiles_extent>("tiles");

    // Logical work domain: one domain point per payload copy.
    auto tile = p.domain<tile_domain>("tile", tiles);

    // Virtual communication endpoints. Target configuration maps them to
    // backend-native ranks or lanes.
    auto producer = p.participant<producer_lane>("producer");
    auto consumer = p.participant<consumer_lane>("consumer");

    // Typed runtime resources. Values are supplied by exec.bind<slot>(...).
    auto workspace = p.resource<workspace_slot, event_copy_workspace>("workspace");
    auto events = p.resource<event_storage_slot, megacu::event_storage_view>("events");

    // Logical event family: one ready event per tile point from producer to
    // consumer, backed by the event storage resource.
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

  // Bind caller-provided storage to resource slots declared in describe(...).
  exec.bind<workspace_slot>(workspace);
  exec.bind<event_storage_slot>(events);

  // Supply the runtime size of the tile domain and launch the lowered target.
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

For `cuda_nvshmem_event_copy`, the workspace is deliberately small:

- `payload`: the buffer written by producer tasks and checked by consumer tasks;
- `scratch`: optional temporary storage for backend or kernel bookkeeping.

The workspace is not where Megacu stores event state. Event state lives in the
separate `event_storage_view` so the design can reason about payload storage
and synchronization storage independently.

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

auto ready = p.event<ready_event>(
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

For `cuda_nvshmem_event_copy`, `ready_event` means:

- for each tile point, producer releases one ready event after writing payload;
- for the same tile point, consumer acquires that ready event before reading
  payload;
- the event-storage resource gives the backend adapter memory where concrete
  signal/wait state can live;
- the event label `"ready"` may appear in logs or metadata dumps, but kernels
  resolve the event through `ctx.event<ready_event>(tile, peer)`.

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

auto tiles = p.extent<tiles_extent>("tiles");
auto tile = p.domain<tile_domain>("tile", tiles);
p.submit(ops::write_then_signal{}, megacu::over(tile), args);
```

The domain name is for authoring and diagnostics. Lowering may replace it with
compact ids in generated metadata, but those ids are not public authoring API.

The first implementation only needs one-dimensional domains. Multi-dimensional
helpers can be added later only if examples need them.

In `cuda_nvshmem_event_copy`, the domain relation is:

```text
tiles_extent = runtime count, for example 128
tile_domain  = logical set {0, 1, ..., 127}
tile point   = one element of tile_domain, for example tile 17
```

The dispatcher decides how tile points map to CUDA blocks, persistent-worker
lanes, ranks, or other backend-native execution coordinates. User code should
not assume `tile 17` means CUDA block 17 or rank 17.

## Virtual Participants

Communication often needs "the peer task" before the backend rank or lane is
known. Megacu models that through virtual participants.

Initial API shape:

```cpp
struct producer_lane;
struct consumer_lane;

auto producer = p.participant<producer_lane>("producer");
auto consumer = p.participant<consumer_lane>("consumer");
```

Virtual participants are declarations of logical roles, not placement
configuration. They are inputs to the dispatcher. The dispatcher selected for a
target decides how those roles map to backend-native ranks, lanes, CTAs, or
peers for each domain point.

For the first static example, a simple dispatcher policy can decide:

- `producer_lane` maps to the source PE for a tile;
- `consumer_lane` maps to the destination PE for the same tile;
- the concrete source/destination relation is represented in the dispatcher's
  participant table, not in CMake syntax and not in kernel code.

CMake should only select or link the dispatcher component that owns this rule.
Runtime kernels do not use participant names as strings.

Virtual participants are allowed only when they remove raw rank/worker ids from
the public program. They must not become a second scheduler API.

## Tasks And Submissions

The public task concept is `submit`: one named op over one logical workset with
typed arguments and dependencies.

Initial API shape:

```cpp
p.submit(
    ops::write_then_signal{},
    megacu::over(tile),
    megacu::place(producer),
    megacu::args()
        .payload(workspace.payload)
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
