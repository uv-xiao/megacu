# Examples

These examples are written against the active lifecycle:

`authored orchestrate program -> CMake target -> parameterized orchestrate call`

The examples are source-informed but not copied implementations. They are meant
to define what Megacu must make implementable.

## Example 1: GEMM + AllReduce Fusion Kernel

This is the first serious validation example. It mirrors the useful shape in
Triton-distributed's `gemm_allreduce.py`: GEMM workers produce output tiles,
communication workers wait for tile readiness, then reduce the produced tiles
across ranks with NVSHMEM/multimem-style primitives.

Program intent:

```text
per rank:
  local partial tile = A_rank x B_rank
  signal tile_ready(tile)

for each output tile:
  wait for all ranks' tile_ready(tile)
  allreduce the tile across ranks
  write final C tile
```

The authored program uses a logical tile domain. The dispatcher decides how
tile points are split between compute and communication CTAs; the program does
not expose CUDA block ids, rank ids, or scheduler lanes.

```cpp
struct m_tiles_extent;
struct n_tiles_extent;
struct output_tile_domain;
struct rank_domain;
struct compute_lane;
struct reduce_lane;
struct partial_ready_event;
struct gemm_ar_workspace_slot;
struct event_storage_slot;

struct gemm_allreduce_program {
  static void describe(megacu::program_builder &p) {
    auto m_tiles = p.extent<m_tiles_extent>("m_tiles");
    auto n_tiles = p.extent<n_tiles_extent>("n_tiles");
    auto tile = p.domain<output_tile_domain>("output_tile", m_tiles, n_tiles);
    auto rank = p.domain<rank_domain>("rank", p.backend_extent("team_size"));

    auto compute = p.participant<compute_lane>("compute");
    auto reduce = p.participant<reduce_lane>("reduce");

    auto ws = p.resource<gemm_ar_workspace_slot, gemm_ar_workspace>("workspace");
    auto events =
        p.resource<event_storage_slot, megacu::event_storage_view>("events");

    auto partial_ready = p.event<partial_ready_event>(
        "partial_ready",
        megacu::over(tile, rank),
        megacu::remote_event{
            .from = compute,
            .to = reduce,
            .storage = events,
            .scope = megacu::memory_scope::device});

    p.submit(
        ops::gemm_tile_produce{},
        megacu::over(tile),
        megacu::place(compute),
        megacu::args()
            .a(ws.a)
            .b(ws.b)
            .partial(ws.partial)
            .release(partial_ready.release()));

    p.submit(
        ops::allreduce_tile_consume{},
        megacu::over(tile),
        megacu::place(reduce),
        megacu::args()
            .partial(ws.partial)
            .out(ws.c)
            .acquire(partial_ready.acquire_all(rank)));
  }
};
```

The compiled target ABI is explicit:

```cpp
void cuda_nvshmem_gemm_allreduce_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::nvshmem_team_view team,
    gemm_ar_problem problem);
```

`gemm_ar_problem` contains dynamic values such as `M`, `N`, `K`, strides, and
tile sizes. These values fill lowered extent and parameter slots inside the
compiled target; they do not select a new dispatcher, scheduler, backend, or
kernel-lowering strategy.

## Example 2: Native CUDA Operator Bodies

The user writes or links native CUDA/C++ operator implementations. Megacu only
provides the lowered context and backend hooks needed to resolve logical
events, participants, and symmetric peer storage.

```cpp
extern "C" __global__
void gemm_tile_produce_kernel(
    megacu::cuda::kernel_context ctx,
    gemm_ar_workspace ws,
    gemm_ar_problem problem) {
  auto tile = ctx.domain_point<output_tile_domain>();
  auto rank = ctx.local_rank();
  auto ready = ctx.event<partial_ready_event>(tile, rank);

  gemm_tile_accumulate(ws.a, ws.b, ws.partial, problem, tile);

  if (threadIdx.x == 0) {
    megacu::nvshmem::signal(ctx, ready, 1);
  }
}

extern "C" __global__
void allreduce_tile_consume_kernel(
    megacu::cuda::kernel_context ctx,
    gemm_ar_workspace ws,
    gemm_ar_problem problem) {
  auto tile = ctx.domain_point<output_tile_domain>();

  for (auto peer : ctx.team<rank_domain>()) {
    auto ready = ctx.event<partial_ready_event>(tile, peer);
    megacu::nvshmem::wait(ctx, ready, 1);
  }

  if (ctx.backend().supports(megacu::capability::multimem_reduce)) {
    megacu::nvshmem::multimem_reduce_tile(ctx, ws.partial, ws.c, problem, tile);
  } else {
    megacu::nvshmem::load_reduce_store_tile(ctx, ws.partial, ws.c, problem, tile);
  }
}
```

This is the critical implementation contract:

- `gemm_tile_accumulate` may be handwritten CUDA, CUTLASS/CuTe code, or another
  native library entrypoint linked into the target.
- `signal`, `wait`, `multimem_reduce_tile`, and `load_reduce_store_tile` are
  backend-provided device primitives linked by the selected backend adapter.
- kernel lowering may stitch the two named ops into one persistent kernel, but
  it must not generate new CUDA source for the program as the normal path.
- no device code parses `"partial_ready"`, `"compute"`, or `"reduce"`; the
  lowered context resolves typed tags to backend-native addresses and peers.

## Example 3: CMake Target

```cmake
megacu_add_components(
  NAME cuda_nvshmem_static
  DISPATCHER tiled_compute_comm_dispatch
  SCHEDULER static_persistent
  KERNEL_LOWERING persistent_stitch
  PLATFORM cuda
  BACKEND nvshmem
)

megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce
  PROGRAM gemm_allreduce_program
  SOURCES gemm_allreduce_orchestrate.cc
  KERNELS gemm_allreduce_kernels.cu
  OPS
    gemm_tile_produce=gemm_tile_produce_kernel
    allreduce_tile_consume=allreduce_tile_consume_kernel
  COMPONENTS cuda_nvshmem_static
)
```

CMake selects and links the reusable dispatcher, scheduler, lowering, platform,
backend, and kernel implementations. The dispatcher owns the mapping from
`compute_lane` and `reduce_lane` to execution placements and backend peers. The
CMake file does not contain ad hoc rank mapping rules.

## Example 4: What Runs

For `cuda_nvshmem_gemm_allreduce_orchestrate`, the built target runs this
sequence:

1. runtime C++ calls
   `cuda_nvshmem_gemm_allreduce_orchestrate(workspace, events, team, problem)`;
2. the compiled function validates the problem envelope and fills lowered
   resource, extent, and backend handle slots;
3. internal `run(...)` launches the linked CUDA/NVSHMEM execution path;
4. the dispatcher payload maps `output_tile_domain` points to compute and
   communication placements;
5. GEMM workers compute partial output tiles and release `partial_ready_event`;
6. communication workers acquire the per-rank tile events, reduce the tile
   through the linked NVSHMEM/multimem or load-reduce-store primitive, and write
   the final output tile.

Term meanings in this example:

- `output_tile`: logical `(m_tile, n_tile)` work item for the output matrix.
- `rank`: logical team participant domain, filled from the backend team size.
- `compute_lane`: virtual participant role for GEMM-producing work.
- `reduce_lane`: virtual participant role for communication/reduction work.
- `workspace`: caller-owned tensor, partial, output, and scratch views.
- `events`: caller-owned synchronization storage for per-tile readiness.
- `problem`: runtime matrix shape, strides, datatype envelope, and tile shape.

## Example 5: Framework Wrapper Path

Megacu stays C++/CUDA. A framework wrapper owns tensor conversion and workspace
policy, then calls the compiled target directly.

```cpp
void torch_gemm_allreduce(
    torch_tensor a,
    torch_tensor b,
    torch_tensor c,
    torch_tensor partial,
    torch_tensor event_buffer,
    dist_handle handle,
    gemm_ar_problem problem) {
  cuda_nvshmem_gemm_allreduce_orchestrate(
      gemm_ar_workspace{
          .a = wrapper_tensor(a),
          .b = wrapper_tensor(b),
          .partial = wrapper_tensor(partial),
          .c = wrapper_tensor(c)},
      wrapper_events(event_buffer),
      wrapper_nvshmem(handle),
      problem);
}
```

The wrapper may cache workspace allocation or event buffers. It does not choose
the scheduler or backend at runtime.

## Example 6: Larger MPK-Style CUDA Serving Program

The second example is larger and intentionally closer to MPK. MPK's repository
also contains older Mirage code, so this example only uses MPK-related evidence
from `research/repos/mirage-mpk/src/kernel/` and
`research/repos/mirage-mpk/python/mirage/mpk/`.

Those MPK paths contain CUDA-provided task implementations, persistent-kernel
construction, and task registration for operator families such as:

- Hopper linear and linear-with-residual tasks;
- Hopper paged attention, RMSNorm, SiLU multiply, embedding, and MoE tasks;
- SM100 MLA decode/reduce/prefill/MTP tasks;
- NVSHMEM allgather and tile-allreduce tasks.

Megacu should be able to express the same kind of serving iteration without
copying MPK's generated-CUDA architecture.

```cpp
struct token_domain;
struct layer_domain;
struct kv_split_domain;
struct norm_ready_event;
struct qkv_ready_event;
struct attention_ready_event;
struct reduce_ready_event;
struct decode_workspace_slot;
struct decode_state_slot;

struct qwen_decode_layer_program {
  static void describe(megacu::program_builder &p) {
    auto tokens = p.extent<tokens_extent>("tokens");
    auto kv_splits = p.extent<kv_splits_extent>("kv_splits");
    auto token = p.domain<token_domain>("token", tokens);
    auto split = p.domain<kv_split_domain>("kv_split", kv_splits);

    auto compute = p.participant<compute_lane>("compute");
    auto comm = p.participant<reduce_lane>("comm");

    auto ws =
        p.resource<decode_workspace_slot, decode_workspace>("workspace");
    auto state =
        p.resource<decode_state_slot, decode_runtime_state>("decode_state");
    auto events =
        p.resource<event_storage_slot, megacu::event_storage_view>("events");

    auto norm_ready =
        p.event<norm_ready_event>("norm_ready", token, {.storage = events});
    auto qkv_ready =
        p.event<qkv_ready_event>("qkv_ready", token, {.storage = events});
    auto attention_ready = p.event<attention_ready_event>(
        "attention_ready", megacu::over(token, split), {.storage = events});
    auto reduce_ready =
        p.event<reduce_ready_event>("reduce_ready", token, {.storage = events});

    p.submit(ops::rmsnorm{}, megacu::over(token), megacu::place(compute),
             megacu::args().x(ws.hidden).scale(ws.norm_weight)
                 .release(norm_ready.release()));

    p.submit(ops::qkv_linear{}, megacu::over(token), megacu::place(compute),
             megacu::args().x(ws.hidden).weight(ws.qkv_weight)
                 .acquire(norm_ready.acquire())
                 .release(qkv_ready.release()));

    p.submit(ops::paged_attention_split{}, megacu::over(token, split),
             megacu::place(compute),
             megacu::args().qkv(ws.qkv).kv_cache(ws.kv_cache).state(state)
                 .acquire(qkv_ready.acquire())
                 .release(attention_ready.release()));

    p.submit(ops::attention_split_reduce{}, megacu::over(token),
             megacu::place(comm),
             megacu::args().partials(ws.attn_partials).out(ws.attn_out)
                 .acquire(attention_ready.acquire_all(split))
                 .release(reduce_ready.release()));

    p.submit(ops::output_linear_residual{}, megacu::over(token),
             megacu::place(compute),
             megacu::args().x(ws.attn_out).weight(ws.out_weight)
                 .residual(ws.hidden).out(ws.next_hidden)
                 .acquire(reduce_ready.acquire()));
  }
};
```

For this MPK-style target:

- CUDA operator bodies are provided in user or backend source files, not emitted
  by Megacu lowering.
- Target lowering links existing CUDA implementations for RMSNorm, linear,
  paged attention, split reduction, and residual output.
- Dynamic serving state such as sequence length, KV page tables, request ids,
  and token counts is a typed runtime resource (`decode_runtime_state`), not a
  hidden field in Megacu's generic core.
- Static scheduling remains the default for the proof. A scheduler-CTA or
  queue-backed strategy is a different linked scheduler component with explicit
  cost, not a different public program API.

This example is bigger than the first proof, but it has the same contract:
Megacu compiles and links native components, materializes metadata, and runs the
linked target. It does not translate the program into new CUDA source.
