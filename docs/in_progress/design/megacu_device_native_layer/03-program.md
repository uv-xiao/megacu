# Program Model

The authored unit in Megacu is now an orchestrate program written in C++.

That program is the logical orchestration description that CMake will compile
and link into a concrete target. It is not a plugin artifact, and it is not a
generic runtime-loaded module.

## What The Orchestrate Program Must Express

The orchestrate program should express only the information needed before
build-graph lowering:

- named ops
- resources
- explicit events
- logical work domains
- orchestration order and control flow
- optional mapping hints for the dispatcher

That is the whole user-visible model.

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

## How Users Author Orchestrate Programs

The intended authoring flow is API collection, not trait filling and not raw
descriptor construction.

The concrete authoring surface should be ordinary C++ orchestration code that
is compiled by CMake. The `program` is a semantic model, not a promise that
runtime C++ must expose a build-capable API.

Shape:

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

The exact syntax is not fixed, but the structure is:

- author one orchestrate function/program
- attach resources/events/domains through small builders
- let the build graph lower the internals later

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

## What The Program Does Not Contain

The orchestrate program should not contain:

- scheduler implementation choice
- kernel-shape choice
- backend/platform selection
- raw packed descriptors
- final worker/rank/resource indices

Those belong to build-graph lowering and to target internals.
