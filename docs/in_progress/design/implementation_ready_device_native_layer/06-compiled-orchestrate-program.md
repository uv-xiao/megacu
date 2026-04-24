# Compiled Orchestrate Program

The primary runtime story should be simple:

- author an orchestrate program in C++
- let CMake compile/link it against Megacu component targets
- call that compiled program directly

That is the normal path. It is not a plugin loader story.

## Direct Call Surface

The normal runtime surface should look like:

```cpp
struct event_copy_program;
struct workspace_slot;
struct event_storage_slot;
struct tiles_extent;

// Ordinary function exported by the compiled orchestrate target.
// It is the user/framework call surface.
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

The descriptor compiled into that target looks like:

```cpp
struct tile_domain;
struct producer_lane;
struct consumer_lane;
struct ready_event;

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
```

The important properties are:

- no string path loading
- no generic `runtime_env`
- no generic resource lookup by string
- no event lookup by string
- no separate generated wrapper required for the primary API
- explicit typed arguments in the function signature
- explicit typed tags for domains, participants, events, and ops

## Why This Is Better

A generic runtime loader weakens the zero-overhead story because it:

- hides what resources are actually required
- shifts mistakes toward runtime checking
- makes the primary API look like a plugin system

The compiled-orchestrate-program path is better because it:

- makes requirements explicit in ordinary C++ signatures
- lets normal code use ordinary CMake linking
- keeps the orchestration code close to the real resources it needs
- gives lowering typed identities for event and participant resolution

## What Runs

The compiled target contains both the user-authored orchestrate function and
Megacu-generated or Megacu-linked target metadata.

At runtime:

1. deployment code calls `cuda_nvshmem_event_copy_orchestrate(...)`;
2. the executor receives explicit runtime values such as workspace views,
   event storage, and `nvshmem_team_view`;
3. `bind` and `run` attach those values to pre-lowered slots and extents;
4. `run` enters the selected CUDA/NVSHMEM execution path;
5. kernels use `kernel_context` to resolve current domain points, virtual
   participants, and event endpoints.

This is still direct C++ execution. The target does not load itself by path and
does not choose a new backend or schedule at runtime.

## Internal Lowering And Metadata

The build graph may still emit internal prepared data such as:

- lowered submission records
- resource/view tables
- event wait/signal tables
- participant mapping tables
- dispatcher payload
- scheduler payload
- kernel launch payload
- backend/platform payload

Those are target internals. They are not the normal user authoring surface.

## Optional Advanced Deployment Path

If the project later needs plugin-style deployment or late-bound artifacts, that
can exist as an optional lower-layer ABI. It should not be the primary design
path or the primary example path.
