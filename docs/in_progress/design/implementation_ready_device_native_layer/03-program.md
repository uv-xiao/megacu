# Program Model

The authored unit in Megacu is now an orchestrate program written in C++.

That program is the logical orchestration description that CMake will compile
and link into a concrete target. It is not a plugin artifact, and it is not a
generic runtime-loaded module.

## Terms

- **Orchestrate program**: the C++ descriptor users write to declare tasks,
  events, resources, and logical extents. CMake lowers it into a compiled,
  parameterized orchestrate function.
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
- **Extent**: the runtime size of a domain. In the first example, `problem.M`,
  `problem.N`, and the tile shape determine the output-tile extents for the
  current call.
- **Resource slot**: a typed program-declared input to the compiled target.
  `gemm_ar_workspace_slot` and `event_storage_slot` are not allocation ids; they
  are internal binding points that the compiled orchestrate function fills from
  its typed parameters.

Names are labels. They are useful for diagnostics and materialized metadata, but
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
| Resource slot | Lets lowering connect descriptor resources to typed function parameters. | The implementation would need a generic `runtime_env`, brittle positional convention, or string lookup. |
| Workspace | Groups caller-owned payload/scratch storage without making Megacu an allocator. | Users would either pass many unrelated buffers into every API or Megacu would need to own allocation policy. |
| Event storage | Separates synchronization storage from payload workspace. | Backend event state would be hidden inside payload buffers or allocated by Megacu, weakening explicit resource ownership. |
| Event | Represents a logical wait/signal dependency between tasks. | Communication ordering would live only in kernel code, so the scheduler/lowering could not inspect or validate it. |
| Virtual participant | Names logical communication endpoints before backend rank/lane resolution. | The public program would expose raw ranks/workers, or kernels would hard-code backend placement. |
| Kernel context | Carries lowered domain, participant, event, resource, and backend metadata to device code. | Kernels would need raw backend handles, string lookup, or handwritten per-target glue parameters. |

The removable parts are labels such as `"tile"`, `"ready"`, `"producer"`, and
`"workspace"`. They are not semantic requirements; they exist for diagnostics
and materialized metadata. The required identities are the typed tags and lowered
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
- `include/megacu/views.h`: typed resource views.
- `include/megacu/platform/cuda.h`: first CUDA launch/runtime view.
- `include/megacu/backends/nvshmem.h`: first backend runtime and device views.

## Builder Output

The first implementation should make `program_builder` collect one concrete
host-side record tree. This record tree is internal, but it must be simple
enough that an implementer can write it directly.

Planned internal path: `include/megacu/detail/program_ir.h`.

```cpp
namespace megacu::detail {
using slot_index = std::uint16_t;

enum class extent_source : std::uint8_t {
  runtime_problem_field,
  backend_team_size,
  constant
};

struct extent_decl {
  slot_index slot;
  std::string_view label;
  extent_source source;
  std::string_view source_name;
};

struct domain_decl {
  slot_index slot;
  std::string_view label;
  std::span<const slot_index> extent_slots;
};

struct participant_decl {
  slot_index slot;
  std::string_view label;
};

struct resource_decl {
  slot_index slot;
  std::string_view label;
  std::type_index view_type;
};

struct event_decl {
  slot_index slot;
  std::string_view label;
  std::span<const slot_index> domain_slots;
  slot_index from_participant;
  slot_index to_participant;
  slot_index storage_resource;
  memory_scope scope;
};

enum class event_use_kind : std::uint8_t { acquire_one, acquire_all, release };

enum class event_wait_mode : std::uint8_t {
  none,
  blocking_device_wait,
  phase_satisfied
};

struct event_use {
  slot_index event_slot;
  event_use_kind kind;
  slot_index over_domain_slot;
  event_wait_mode wait_mode;
};

struct arg_binding {
  std::string_view name;
  slot_index resource_slot;
  std::string_view member_name;
};

struct submission_decl {
  slot_index op_slot;
  std::string_view op_name;
  slot_index work_domain_slot;
  slot_index participant_slot;
  std::uint16_t phase;
  std::span<const arg_binding> args;
  std::span<const event_use> events;
};

struct program_ir {
  std::span<const extent_decl> extents;
  std::span<const domain_decl> domains;
  std::span<const participant_decl> participants;
  std::span<const resource_decl> resources;
  std::span<const event_decl> events;
  std::span<const submission_decl> submissions;
};
}
```

Slots are assigned in descriptor traversal order and are stable inside one
orchestrate target. They are not stable across targets and must not appear in
the public API.

The first implementation can use owning `std::vector` storage behind the spans.
The important contract is the field set above, not the container choice.

`event_wait_mode` is not a public scheduler API. It records whether a lowered
event acquire may block in device code. The first builder can infer it from the
event argument helper used by the target or from a CMake op requirement. Target
lowering needs this bit so the overlap scheduler can reject blocking waits that
do not have a co-residency or completed-phase proof.

## First Implementation Shape

To make build-time lowering concrete, the first implementation should not try
to inspect arbitrary C++ function bodies. Users write a C++ program descriptor
with an explicit `describe(...)` method. CMake lowers that descriptor into an
ordinary parameterized orchestrate function.

For the first validation target family:

- `problem` is a runtime descriptor containing `M`, `N`, `K`, strides, datatype
  envelope, and tile shape.
- `output_tile` is the logical domain over GEMM output tiles. If `M` and `N`
  imply 128 output tiles, the domain has points `tile 0` through `tile 127`.
- a GEMM producer task and an AllReduce consumer task are submitted for each
  output tile.
- `workspace` is caller-provided storage containing `A`, `B`, per-rank partial
  output, final output, and scratch buffers used by those tasks.
- `events` is caller-provided event storage where the backend adapter can place
  the concrete signal/wait state for `partial_ready_event`.
- the compiled function parameter `workspace` fills the descriptor's
  `gemm_ar_workspace_slot`.
- the compiled function parameter `events` fills the descriptor's
  `event_storage_slot`.
- the compiled function parameter `problem` fills the matrix and tile extents
  and determines how many logical output tile points run.

```cpp
struct gemm_allreduce_overlap_program {
  static void describe(megacu::program_builder &p) {
    // Runtime values supplied by the compiled function's `problem` parameter.
    auto m_tiles = p.extent<m_tiles_extent>("m_tiles");
    auto n_tiles = p.extent<n_tiles_extent>("n_tiles");

    // Logical work domain: one domain point per output matrix tile.
    auto tile = p.domain<output_tile_domain>("output_tile", m_tiles, n_tiles);
    auto rank = p.domain<rank_domain>("rank", p.backend_extent("team_size"));

    // Virtual execution roles. The selected dispatcher maps them to
    // backend-native CTAs, lanes, ranks, or peers during target lowering.
    auto compute = p.participant<compute_lane>("compute");
    auto reduce = p.participant<reduce_lane>("reduce");

    // Typed runtime resources. Values are supplied by compiled function
    // parameters with matching slots.
    auto workspace = p.resource<gemm_ar_workspace_slot, gemm_ar_workspace>("workspace");
    auto events = p.resource<event_storage_slot, megacu::event_storage_view>("events");

    // Logical event family: one partial-ready event per tile and rank.
    auto partial_ready = p.event<partial_ready_event>(
        "partial_ready",
        megacu::over(tile, rank),
        megacu::remote_event{.from = compute, .to = reduce, .storage = events});

    p.submit(
        ops::gemm_tile_produce{},
        megacu::over(tile),
        megacu::place(compute),
        megacu::args()
            .a(workspace.a)
            .b(workspace.b)
            .partial(workspace.partial)
            .release(partial_ready.release()));

    p.submit(
        ops::allreduce_tile_consume{},
        megacu::over(tile),
        megacu::place(reduce),
        megacu::args()
            .partial(workspace.partial)
            .out(workspace.c)
            .acquire(partial_ready.acquire_all(rank)));
  }
};

megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
```

`describe(...)` is the concrete program surface that CMake/native build tooling
can compile into metadata. The parameterized
`cuda_nvshmem_gemm_allreduce_overlap_orchestrate` function is the direct call surface
used by applications and framework wrappers. This keeps build work out of
runtime C++ while avoiding a hidden parser for arbitrary C++.

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
struct gemm_tile_produce {
  static constexpr auto name = "gemm_tile_produce";
};

struct allreduce_tile_consume {
  static constexpr auto name = "allreduce_tile_consume";
};
}
```

A named op is a type or symbol known to the native build graph. The user should
not construct an op descriptor manually.

The CMake target maps op tags to implementation symbols:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_overlap
  PROGRAM gemm_allreduce_overlap_program
  SOURCES gemm_allreduce_orchestrate.cc
  KERNELS gemm_allreduce_kernels.cu
  OPS
    gemm_tile_produce=gemm_tile_produce_kernel
    allreduce_tile_consume=allreduce_tile_consume_kernel
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
enum class status_code : std::uint8_t {
  ok,
  invalid_argument,
  backend_error
};
struct status {
  status_code code;
  std::string_view message;
};
struct backend_id {
  std::uint16_t value;
};
struct session_id {
  std::uint64_t value;
};
struct symmetric_buffer_view {
  void *data;
  std::int64_t bytes;
  backend_id backend;
  session_id session;
};
struct event_storage_view {
  symmetric_buffer_view buffer;
};
struct tensor_view {
  void *data;
  std::int64_t bytes;
  dtype type;
};
struct symmetric_tensor_view {
  symmetric_buffer_view buffer;
  dtype type;
};
}
```

Resource views are explicit C++ arguments to the orchestrate function. There is
no generic resource map, no string lookup, and no public resource id plumbing in
normal examples.

`workspace_view` is not magic global memory. It is a typed view over caller-
provided storage:

```cpp
struct gemm_ar_workspace {
  megacu::tensor_view a;
  megacu::tensor_view b;
  megacu::symmetric_tensor_view partial;
  megacu::tensor_view c;
  megacu::span<std::byte> scratch;
};
```

The orchestrate program receives this view, passes typed slices to submissions,
and lowering records which resource slots kernels need. Allocation remains with
the caller or framework wrapper.

For `cuda_nvshmem_gemm_allreduce`, the workspace is explicit:

- `a` and `b`: input matrix views;
- `partial`: symmetric or peer-addressable per-rank GEMM output storage;
- `c`: final reduced output;
- `scratch`: optional temporary storage for backend or kernel bookkeeping.

The workspace is not where Megacu stores event state. Event state lives in the
separate `event_storage_view` so the design can reason about payload storage
and synchronization storage independently.

For remote CUDA+NVSHMEM events, `event_storage_view::buffer` must carry
backend/session identity matching the `team_view` passed to the compiled target.
The launch adapter or caller owns allocation; the compiled target owns
validation before launch.

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
struct partial_ready_event;

auto partial_ready = p.event<partial_ready_event>(
    "partial_ready",
    megacu::over(tile, rank),
    megacu::remote_event{
        .from = compute,
        .to = reduce,
        .storage = events});

args()
    .release(partial_ready.release())
    .acquire(partial_ready.acquire_all(rank));
```

Event handles own logical coordination only. Backend signal objects, event
storage offsets, memory-order details, and remote-rank mappings are target
internals owned by lowering and backend adapters.

The event name `"partial_ready"` is diagnostic. The typed tag
`partial_ready_event` is the stable program identity. Lowering turns
`partial_ready_event` into an event slot and an event-storage layout. At
runtime, the backend adapter resolves that slot plus the current domain point
and virtual participants into the concrete NVSHMEM signal address or equivalent
backend object.

For `cuda_nvshmem_gemm_allreduce`, `partial_ready_event` means:

- for each output tile and producing rank, the GEMM producer releases readiness
  after writing its partial tile;
- the AllReduce consumer acquires all rank-specific readiness events before
  reducing that tile;
- the event-storage resource gives the backend adapter memory where concrete
  signal/wait state can live;
- the event label `"partial_ready"` may appear in logs or metadata dumps, but
  kernels resolve the event through `ctx.event<partial_ready_event>(tile, peer)`.

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
struct output_tile_domain;

auto m_tiles = p.extent<m_tiles_extent>("m_tiles");
auto n_tiles = p.extent<n_tiles_extent>("n_tiles");
auto tile = p.domain<output_tile_domain>("output_tile", m_tiles, n_tiles);
p.submit(ops::gemm_tile_produce{}, megacu::over(tile), args);
```

The domain name is for authoring and diagnostics. Lowering may replace it with
compact ids in materialized metadata, but those ids are not public authoring
API.

The first implementation needs a two-dimensional output-tile domain for
GEMM+AllReduce. Multi-dimensional helpers should still be narrow: they exist to
express real tile coordinates, not to expose backend worker ids.

In `cuda_nvshmem_gemm_allreduce`, the domain relation is:

```text
m_tiles_extent     = runtime count derived from M and tile_m
n_tiles_extent     = runtime count derived from N and tile_n
output_tile_domain = logical set {(m, n)}
tile point         = one element, for example output tile (3, 7)
```

The dispatcher decides how tile points map to CUDA blocks, persistent-worker
lanes, ranks, or other backend-native execution coordinates. User code should
not assume `tile 17` means CUDA block 17 or rank 17.

## Virtual Participants

Communication often needs "the peer task" before the backend rank or lane is
known. Megacu models that through virtual participants.

Initial API shape:

```cpp
struct compute_lane;
struct reduce_lane;

auto compute = p.participant<compute_lane>("compute");
auto reduce = p.participant<reduce_lane>("reduce");
```

Virtual participants are declarations of logical roles, not placement
configuration. They are inputs to the dispatcher. The dispatcher selected for a
target decides how those roles map to backend-native ranks, lanes, CTAs, or
peers for each domain point.

For the first static example, a simple dispatcher policy can decide:

- `compute_lane` maps to GEMM-producing placements for an output tile;
- `reduce_lane` maps to communication/reduction placements for the same tile;
- backend peers for each rank-specific partial tile are represented in the
  dispatcher's participant table, not in CMake syntax and not in kernel code.

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
    ops::gemm_tile_produce{},
    megacu::over(tile),
    megacu::place(compute),
    megacu::args()
        .a(workspace.a)
        .b(workspace.b)
        .partial(workspace.partial)
        .release(partial_ready.release()));
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
struct m_tiles_extent;
struct n_tiles_extent;
struct output_tile_domain;
struct compute_lane;
struct reduce_lane;
struct partial_ready_event;
struct gemm_ar_workspace_slot;
struct event_storage_slot;

struct gemm_allreduce_overlap_program {
  static void describe(megacu::program_builder &p);
};
```

The concrete structure is:

- author one program descriptor with `describe(...)`
- attach resources/events/domains through small builders
- let the build graph lower the descriptor into target metadata
- expose an ordinary parameterized runtime function whose implementation maps
  parameters to lowered slots and enters the lowered execution path internally

For the first implementation, the required public surface is:

- `megacu::program_builder`
- `megacu::domain`
- `megacu::event`
- `megacu::participant`
- `megacu::over(domain)`
- `megacu::place(participant)`
- `megacu::args()`
- `megacu::program_builder::submit(...)`
- declared or exported parameterized orchestrate function, such as
  `cuda_nvshmem_gemm_allreduce_overlap_orchestrate(...)`

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
void gemm_tile_produce_kernel(
    megacu::cuda::kernel_context ctx,
    gemm_ar_workspace workspace,
    gemm_ar_problem problem) {
  auto tile = ctx.domain_point<output_tile_domain>();
  auto rank = ctx.local_rank();
  auto ready = ctx.event<partial_ready_event>(tile, rank);

  gemm_tile_accumulate(workspace.a, workspace.b, workspace.partial, problem, tile);
  megacu::nvshmem::signal(ctx, ready, 1);
}
```

`ctx.local_rank()` and `ctx.event<partial_ready_event>(...)` are where the
logical rank and event tag resolve to the backend-native endpoint for this
lowered target. There is no device-side string lookup.

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
`examples/cuda_nvshmem_gemm_allreduce/` and compile checks under
`tests/build/`.

Program-model evidence:

- a compile-only example using the public builder APIs above
- no user-authored packed descriptors
- no public task-trait field filling
- no string resource lookup
- no device-side event-name lookup
- materialized metadata showing domain, participant, and event tags resolved to
  compact slots
- direct call of the compiled orchestrate function from runtime C++
