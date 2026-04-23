# Compiled Orchestrate Program

The primary runtime story should be simple:

- author an orchestrate program in C++
- let CMake compile/link it against Megacu component targets
- call that compiled program directly

That is the normal path. It is not a plugin loader story.

## Direct Call Surface

The normal runtime surface should look like:

```cpp
void paged_attention_orchestrate(
    workspace_view workspace,
    event_storage_view events,
    nvshmem_comm_handle nvshmem,
    std::int32_t num_tiles,
    std::int32_t num_tokens) {
  megacu::orchestrator orch{workspace, events, nvshmem};

  auto tile = orch.domain("tile", num_tiles);
  auto ready = orch.event("payload_ready", tile, megacu::remote_event{});

  orch.submit(
      ops::reduce_then_signal,
      over(tile),
      args().out(workspace.partial()).signal(ready.release()));

  orch.submit(
      ops::wait_and_consume,
      over(tile),
      args().wait(ready.acquire()).inp(workspace.partial()).out(workspace.out()));

  orch.run(run_spec{.tokens = num_tokens});
}
```

The important properties are:

- no string path loading
- no generic `runtime_env`
- no generic resource lookup by string
- no separate generated wrapper required for the primary API
- explicit typed arguments in the function signature

## Why This Is Better

A generic runtime loader weakens the zero-overhead story because it:

- hides what resources are actually required
- shifts mistakes toward runtime checking
- makes the primary API look like a plugin system

The compiled-orchestrate-program path is better because it:

- makes requirements explicit in ordinary C++ signatures
- lets normal code use ordinary CMake linking
- keeps the orchestration code close to the real resources it needs

## Internal Lowering And Metadata

The build graph may still emit internal prepared data such as:

- lowered submission records
- resource/view tables
- event wait/signal tables
- dispatcher payload
- scheduler payload
- kernel launch payload
- backend/platform payload

Those are target internals. They are not the normal user authoring surface.

## Optional Advanced Deployment Path

If the project later needs plugin-style deployment or late-bound artifacts, that
can exist as an optional lower-layer ABI. It should not be the primary design
path or the primary example path.
