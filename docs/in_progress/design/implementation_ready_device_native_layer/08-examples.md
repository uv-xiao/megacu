# Examples

These examples are written against the new lifecycle:

`authored orchestrate program -> CMake target -> run`

## Example 1: CUDA Kernel With Backend Primitive Inside

Named kernels remain ordinary CUDA/C++ kernels. Fine-grained overlap can live
inside them.

```cpp
extern "C" __global__
void reduce_then_signal_kernel(float *out, nvshmem_signal_handle sig) {
  // Native CUDA compute.
  reduce_tile(out);

  // Native backend primitive inside the kernel body.
  if (threadIdx.x == 0) {
    nvshmemx_signal_op(sig, 1);
  }
}
```

Feature shown: Megacu does not require a separate public "backend primitive op"
or fragment taxonomy. A named kernel can contain fine-grained backend behavior
directly.

## Example 2: Authored Orchestrate Program

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

Feature shown: the authored object is the orchestrate program itself. There is
no separate primary runtime wrapper to generate or call.

## Example 3: CMake Builds Reusable Components And The Orchestrate Target

```cmake
megacu_add_components(
  NAME cuda_nvshmem_static
  DISPATCHER tile_dispatch
  SCHEDULER static_persistent
  KERNEL_LOWERING persistent_stitch
  PLATFORM cuda
  BACKEND nvshmem
  KERNELS reduce_then_signal.cu wait_and_consume.cu
)

megacu_add_orchestrate_target(
  TARGET paged_attention_orchestrate
  SOURCES paged_attention_orchestrate.cc
  COMPONENTS cuda_nvshmem_static
)
```

Feature shown: CMake owns reusable component compilation and orchestrate-target
compilation/linking. Ordinary user code just calls the compiled orchestration.

## Example 4: External Framework Wrapper Path

```cpp
void torch_paged_attention(
    torch_workspace_view workspace,
    torch_event_view events,
    dist_handle handle,
    std::int32_t num_tiles,
    std::int32_t num_tokens) {
  paged_attention_orchestrate(
      wrapper_workspace(workspace),
      wrapper_events(events),
      wrapper_nvshmem(handle),
      num_tiles,
      num_tokens);
}
```

Feature shown: Megacu itself stays C++ only, while external framework wrappers
can call the compiled orchestrate program directly.
