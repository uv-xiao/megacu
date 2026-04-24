# Examples

These examples are written against the active lifecycle:

`authored orchestrate program -> CMake target -> parameterized orchestrate call`

The examples are source-informed but not copied implementations. They are meant
to define what Megacu must make implementable.

## Example 1: GEMM + AllReduce Variants

The first serious validation family should have two concrete targets with the
same public resource and problem shape:

| Target | Progress model | Purpose |
| --- | --- | --- |
| `cuda_nvshmem_gemm_allreduce_phased` | phased | MPK-style correctness baseline: compute partial tiles, finish the compute phase, then run reduction. |
| `cuda_nvshmem_gemm_allreduce_overlap` | co-resident persistent | overlap proof: GEMM workers and communication workers are live together, tile readiness lets reduction start before all GEMM work is complete. |

Both targets are useful. The phased target is simpler and should be the first
correctness baseline. The overlap target is the performance-oriented proof that
Megacu can express fine-grained compute/communication fusion without generated
CUDA source.

The overlap target mirrors the useful shape in Triton-distributed's
`gemm_allreduce.py`: GEMM workers produce output tiles, communication workers
wait for tile readiness, then reduce the produced tiles across ranks with
NVSHMEM/multimem-style primitives.

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

struct gemm_allreduce_overlap_program {
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
            .scope = megacu::memory_scope::remote_team});

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
            .acquire(partial_ready.acquire_all(
                rank,
                megacu::event_wait::blocking_device())));
  }
};
```

The compiled target ABI is explicit:

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);

megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
```

`gemm_ar_problem` contains dynamic values such as `M`, `N`, `K`, strides, and
tile sizes. These values fill lowered extent and parameter slots inside the
compiled target; they do not select a new dispatcher, scheduler, backend, or
kernel-lowering strategy.

For the phased target, the descriptor may use the same logical event and
submission shape, but the selected scheduler materializes
`progress_model::phased`. For the overlap target, the selected scheduler must
materialize `progress_model::co_resident_persistent` and a residency group that
contains both `compute_lane` and `reduce_lane`.

The overlap target is invalid unless target lowering can prove the communication
guard:

- GEMM producers and AllReduce consumers are in the same persistent launch;
- at least one compute worker and one communication worker are resident for the
  lifetime of the overlapped region;
- communication workers only perform blocking waits on events whose producers
  are in the same residency group or in a completed earlier phase.

This guard is general scheduler metadata, not a GEMM-specific rule.

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

  if (ctx.backend().supports(megacu::cuda::capability::multimem_reduce)) {
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
  TARGET cuda_nvshmem_gemm_allreduce_phased
  PROGRAM gemm_allreduce_phased_program
  SOURCES gemm_allreduce_phased_orchestrate.cc
  KERNELS gemm_allreduce_kernels.cu
  OPS
    gemm_tile_produce
      LAUNCH gemm_tile_produce_kernel
    allreduce_tile_consume
      LAUNCH allreduce_tile_consume_kernel
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE phased
  BACKEND_ENVELOPE
    TEAM_SIZE 2
)

megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_overlap
  PROGRAM gemm_allreduce_overlap_program
  SOURCES gemm_allreduce_overlap_orchestrate.cc
  KERNELS gemm_allreduce_kernels.cu
  OPS
    gemm_tile_produce
      LAUNCH gemm_tile_produce_kernel
      CALLABLE gemm_tile_produce_body
    allreduce_tile_consume
      LAUNCH allreduce_tile_consume_kernel
      CALLABLE allreduce_tile_consume_body
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE co_resident_persistent
  BACKEND_ENVELOPE
    TEAM_SIZE 2
)
```

CMake selects and links the reusable dispatcher, scheduler, lowering, platform,
backend, and kernel implementations. The target envelope fixes the first proof
to a two-PE CUDA+NVSHMEM team, but the dispatcher still owns the mapping from
`compute_lane` and `reduce_lane` to execution placements and backend peers. The
CMake file does not contain ad hoc rank mapping rules such as
`compute_lane -> rank 0`.

`SCHEDULER_MODE` selects a reusable scheduler mode for one target. It is not a
runtime choice. The overlap mode must emit a co-residency guard in
the schedule metadata section; if it cannot, the build fails.

`LAUNCH` names a launchable kernel entrypoint. `CALLABLE` is only listed for the
overlap target because the selected `persistent_stitch` lowering may need
device-callable bodies. A CUDA target that keeps each op as a launchable kernel
does not need `CALLABLE`.

## Example 4: What Runs

For `cuda_nvshmem_gemm_allreduce_phased_orchestrate`, the built target runs this
sequence:

1. runtime C++ calls the compiled target with explicit `workspace`, `events`,
   `launch`, `team`, and `problem`;
2. the compiled function validates the problem envelope and fills lowered slots;
3. internal `run(...)` computes all local partial tiles for the current phase;
4. the phase boundary guarantees producers have completed;
5. reduction workers reduce tiles through the linked NVSHMEM/multimem or
   load-reduce-store primitive and write final output tiles.

For `cuda_nvshmem_gemm_allreduce_overlap_orchestrate`, the built target runs
this sequence:

1. runtime C++ calls
   `cuda_nvshmem_gemm_allreduce_overlap_orchestrate(workspace, events, launch, team, problem)`;
2. the compiled function validates the problem envelope and fills lowered
   resource, extent, CUDA launch, and backend handle slots;
3. internal `run(...)` launches one co-resident persistent CUDA/NVSHMEM execution
   path with compute and communication workers in the same residency group;
4. the dispatcher payload maps `output_tile_domain` points to compute and
   communication placements;
5. GEMM workers compute partial output tiles and release `partial_ready_event`;
6. communication workers acquire the per-rank tile events, reduce the tile
   through the linked NVSHMEM/multimem or load-reduce-store primitive, and write
   the final output tile.

The overlap sequence is valid only because the schedule metadata contains a
co-residency guard. A blocking communication worker must never depend on a GEMM
worker that might be a non-resident CUDA block waiting behind it.

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
policy, a launch adapter owns process/rank/bootstrap state, then the wrapper
calls the compiled target directly.

```cpp
megacu::status torch_gemm_allreduce(
    torch_tensor a,
    torch_tensor b,
    torch_tensor c,
    megacu_torch::symmetric_tensor partial,
    megacu_torch::event_tensor events,
    megacu_torch::nvshmem_session &session,
    gemm_ar_problem problem) {
  auto launch = session.current_cuda_launch_view();
  auto team = session.team_view();

  return cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
      gemm_ar_workspace{
          .a = wrapper_tensor(a),
          .b = wrapper_tensor(b),
          .partial = wrapper_symmetric_tensor(partial),
          .c = wrapper_tensor(c)},
      wrapper_event_storage(events),
      launch,
      team,
      problem);
}
```

The wrapper may cache workspace allocation or event buffers. It does not choose
the scheduler or backend at runtime. It must check that `partial` and `events`
come from the same NVSHMEM session and are symmetric allocations. The wrapper
returns the compiled target `megacu::status`; a Python binding may translate a
non-OK status into a Python exception at the framework boundary.

An external Torch wrapper or integration test can expose a Python-facing shape
like this without making Python part of Megacu core:

```python
session = megacu.torch.NvshmemSession.from_torch_distributed()
partial = session.empty_symmetric_like(c)
events = session.empty_event_storage(
    cuda_nvshmem_gemm_allreduce.event_bytes(problem))

megacu.torch.gemm_allreduce(
    a, b, c, partial=partial, events=events,
    session=session, problem=problem)
```

Run it with a fixed-size Torch world:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  tests/integration/torch/test_cuda_nvshmem_gemm_allreduce.py
```

The first MPI launch path should look like:

```cpp
auto session = megacu::launch::mpi_nvshmem_session::create({
    .comm = MPI_COMM_WORLD,
    .local_cuda_device = local_device_from_mpi_rank(),
    .initialize_mpi = false});

auto launch = session.launch(stream);
auto team = session.team();
megacu::symmetric_buffer_view partial_buffer =
    session.symmetric_alloc(partial_bytes, 128);
megacu::symmetric_buffer_view event_buffer =
    session.symmetric_alloc(event_bytes, alignof(std::uint64_t));

gemm_ar_workspace workspace{a, b, typed_symmetric(partial_buffer, dtype), c,
                            scratch};
megacu::event_storage_view events{event_buffer};

auto status = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    workspace, events, launch, team, problem);
```

Run it with:

```bash
mpirun -np 2 ./cuda_nvshmem_gemm_allreduce_mpi --m 128 --n 128 --k 128
```

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
