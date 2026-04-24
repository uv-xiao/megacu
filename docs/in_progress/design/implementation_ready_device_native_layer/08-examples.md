# Examples

These examples are written against the new lifecycle:

`authored orchestrate program -> CMake target -> run`

## Example 1: CUDA Kernel With Backend Primitive Inside

Named kernels remain ordinary CUDA/C++ kernels, but Megacu passes a lowered
kernel context when the kernel needs Megacu-managed event or peer resolution.
Fine-grained overlap can live inside the kernel.

```cpp
extern "C" __global__
void write_then_signal_kernel(
    megacu::cuda::kernel_context ctx,
    event_copy_workspace workspace) {
  auto tile = ctx.domain_point<tile_domain>();
  auto peer = ctx.peer<consumer_lane>(tile);
  auto ready = ctx.event<ready_event>(tile, peer);

  // Native CUDA compute.
  workspace.payload[tile.linear] = make_payload(tile.linear);

  // Native backend primitive inside the kernel body.
  if (threadIdx.x == 0) {
    megacu::nvshmem::signal(ctx, ready, 1);
  }
}
```

Feature shown: Megacu does not require a separate public "backend primitive op"
or fragment taxonomy. A named kernel can contain fine-grained backend behavior
directly, while event and peer resolution still comes from lowered target
metadata rather than raw strings.

## Example 2: Authored Orchestrate Program

```cpp
struct tiles_extent;
struct tile_domain;
struct producer_lane;
struct consumer_lane;
struct ready_event;
struct workspace_slot;
struct event_storage_slot;

struct event_copy_program {
  static void describe(megacu::program_builder &p) {
    // `tiles` is a runtime count supplied by the parameterized orchestrate
    // function.
    auto tiles = p.extent<tiles_extent>("tiles");

    // `tile` is the logical set of work items, one point per payload copy.
    auto tile = p.domain<tile_domain>("tile", tiles);

    // These are virtual endpoints. CMake target metadata maps them to ranks.
    auto producer = p.participant<producer_lane>("producer");
    auto consumer = p.participant<consumer_lane>("consumer");

    // Runtime resources. The parameterized orchestrate function supplies them.
    auto workspace = p.resource<workspace_slot, event_copy_workspace>("workspace");
    auto events = p.resource<event_storage_slot, megacu::event_storage_view>("events");

    // One logical event per tile point, backed by the events resource.
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

Feature shown: the authored descriptor is the orchestrate program. Names such as
`"tile"` and `"ready"` are labels; typed tags such as `tile_domain` and
`ready_event` are lowered into slots.

## Example 3: CMake Builds Reusable Components And The Orchestrate Target

```cmake
megacu_add_components(
  NAME cuda_nvshmem_static
  DISPATCHER tile_dispatch
  SCHEDULER static_persistent
  KERNEL_LOWERING persistent_stitch
  PLATFORM cuda
  BACKEND nvshmem
)

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

Feature shown: CMake owns reusable component compilation and orchestrate-target
compilation/linking. Ordinary user code just calls the compiled orchestration.
The selected dispatcher inside `cuda_nvshmem_static` owns participant placement
and emits the mapping from virtual participants to backend peers for this
target.

## Example 4: External Framework Wrapper Path

```cpp
void torch_event_copy(
    torch_workspace_view workspace,
    torch_event_view events,
    dist_handle handle,
    std::int32_t tiles) {
  cuda_nvshmem_event_copy_orchestrate(
      wrapper_workspace(workspace),
      wrapper_events(events),
      wrapper_nvshmem(handle),
      tiles);
}
```

Feature shown: Megacu itself stays C++ only, while external framework wrappers
can call the compiled orchestrate program directly.

## Example 5: What Runs

For `cuda_nvshmem_event_copy_orchestrate`, the built target runs this sequence:

1. runtime C++ calls
   `cuda_nvshmem_event_copy_orchestrate(workspace, events, team, tiles)`;
2. generated or linked target code binds function parameters to lowered slots;
3. internal fast-path execution launches the selected static persistent
   CUDA/NVSHMEM path;
4. the dispatcher metadata maps each `tile_domain` point to producer and
   consumer backend peers;
5. the producer kernel writes payload and signals `ready_event` through the
   NVSHMEM adapter;
6. the consumer kernel waits on the resolved `ready_event` endpoint and checks
   payload visibility.

No step parses `"ready"` or `"producer"` at runtime.

Term meanings in this example:

- `tiles`: runtime count, such as `128`.
- `tile`: logical domain created from that count, with points `0..127`.
- `workspace`: caller-owned payload/scratch storage used by kernels.
- `events`: caller-owned synchronization storage used by backend event
  endpoints.
- compiled parameter binding: internal target code maps typed function
  parameters to lowered resource and extent slots.
