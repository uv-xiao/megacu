# Megacu Megakernel Research Notes

Date: 2026-04-21

Positioning update, 2026-04-23: this note was written before the user clarified
that Megacu should not be tightly bound to CUDA. Treat the C++/CUDA phrasing
below as historical source-reading context. The active design position is a
platform-neutral device-native core with CUDA/NVSHMEM as the first proof point;
see `docs/in_progress/design/megacu_cpp_cuda_layer.md`.

This note summarizes two closely related systems, Mirage Persistent Kernel (MPK)
and Event Tensor, then proposed the first Megacu direction: a thin C++/CUDA
layer for writing megakernels with CUDA-native performance and no abstraction
tax on the hot path. That performance goal remains, but the core API boundary is
now device-native and platform-neutral rather than CUDA-only.

## Downloaded Material

- MPK paper: `research/papers/mpk-2512.22219.pdf`
- MPK arXiv source: `research/sources/mpk/`
- MPK repository: `research/repos/mirage-mpk`, branch `mpk`, commit `68f4857`
- Event Tensor paper: `research/papers/event-tensor-2604.13327.pdf`
- Event Tensor arXiv source: `research/sources/event-tensor/`
- Event Tensor repository: no public implementation repository found as of
  2026-04-21. I checked arXiv metadata, web search, and GitHub API repository
  search for the title/arXiv id/Event Tensor Compiler. Results were paper
  mirrors/news lists, not an implementation.

Primary sources:

- MPK arXiv: https://arxiv.org/abs/2512.22219
- MPK repo: https://github.com/mirage-project/mirage
- Event Tensor arXiv: https://arxiv.org/abs/2604.13327

## One-Screen Takeaway

MPK and Event Tensor both attack the same bottleneck: modern inference spends too
much time crossing kernel boundaries. A megakernel removes the launch sequence
and replaces kernel-level synchronization with fine-grained dependencies between
SM-sized tasks.

MPK is closest to an end-to-end system: Python/PyTorch-facing API, compiler, task
graph generator, generated CUDA, and an in-kernel runtime with worker and
scheduler CTAs. Its key concrete idea is an SM-level task/event graph. The public
repo confirms this design in code: `TaskDesc`, `EventDesc`, worker queues,
scheduler queues, event counters, NVSHMEM support, and generated task variants.

Event Tensor is closest to a compiler abstraction: it lifts events from
individual synchronization objects into tensor-shaped, symbolic compiler IR
objects. Its most important contribution is not a new semaphore primitive, but a
compact way to describe many fine-grained dependencies, including dynamic shapes
and data-dependent fan-in/fan-out such as MoE routing.

For Megacu, the opportunity is different: do not build a large automatic ML
compiler first. Build the thinnest device-native layer that lets expert kernel
programmers express MPK/Event-Tensor-style task graphs while preserving
platform-native syntax and performance. CUDA is the first and most important
proof point, but not the core abstraction boundary. The sell point should be:
"write a megakernel close to the native platform, keep native performance, pay
only for the synchronization you explicitly ask for."

## MPK Notes

### Problem Framing

MPK argues that the kernel-per-operator model blocks three optimizations:

1. It adds launch overhead across hundreds or thousands of per-token inference
   kernels.
2. It inserts implicit full-kernel barriers, even when downstream work only
   depends on a tile of upstream output.
3. It prevents fine-grained compute/communication overlap, especially around
   collectives such as AllReduce or ReduceScatter.

CUDA Graphs reduce launch overhead, but they still operate at kernel granularity.
MPK instead makes the whole model run inside a persistent kernel.

### Core Abstraction: SM-Level Task Graph

MPK decomposes operators into tasks intended to run on one SM. Tasks produce and
consume tensor regions. Events sit between producer tasks and consumer tasks:

```text
producer task -> event counter -> consumer task(s)
```

The graph is finer than a CUDA Graph. Instead of "kernel A before kernel B",
MPK can encode "tile A[i] before tile B[i]". This allows downstream tasks to
start as soon as their precise input tile is ready.

In the repo, this maps to:

- `FullTaskDesc` and `TaskDesc` in
  `include/mirage/persistent_kernel/runtime_header.h`
- `EventDesc` with `num_triggers`, `first_task_id`, `last_task_id`
- `all_tasks`, `all_events`, `all_event_counters` in `RuntimeConfig`
- `generate_task_graph()` in `src/kernel/runtime.cc`

The task descriptor is intentionally low-level: task type, variant id, input
and output pointers, one dependent event, one trigger event, and a small metadata
union for expert/request/KV/split information.

### Compiler and Graph Construction

MPK's compiler takes a tensor program and inference configuration, decomposes
operators into SM tasks, analyzes producer-consumer tile overlap, generates
events, and emits a linearized task graph plus CUDA code.

Important details:

- Operator decomposition chooses task grids, usually proportional to SM count.
- Dependency analysis creates events based on overlapping tensor regions between
  producer and consumer task tiles.
- Event fusion merges redundant synchronization nodes.
- Graph normalization ensures tasks have a simple dependency shape, reducing
  runtime descriptor overhead.
- Graph linearization makes tasks launched by an event contiguous, so an event
  stores only `[first_task_id, last_task_id)` rather than a variable-length list.

The repo implementation is more specialized than the paper abstraction. The
current `register_mugraph()` path tracks dependencies mostly between consecutive
customized operators and uses grid mapping metadata to determine partitions.
That is a practical engineering tradeoff: simpler graph generation, less general
than arbitrary DAG event tensor IR.

### In-Kernel Runtime

MPK partitions SMs into workers and schedulers:

- Worker CTAs run task loops. They read task ids from worker queues, copy task
  descriptors into shared memory, wait on dependent events if needed, execute
  the task, and notify trigger events.
- Scheduler warps read activated events from scheduler queues and enqueue ready
  tasks to worker queues.
- Event counters live in device memory and are updated with atomics.
- Remote/multi-GPU events use NVSHMEM paths.

The public runtime has two launch modes:

- Combined `persistent_kernel`, where workers and schedulers are CTAs in one
  kernel.
- Split `worker_kernel` and `scheduler_kernel` launched on streams, which is
  still a persistent execution model but operationally uses two kernels.

Important implementation details:

- Workers prefetch multiple `TaskDesc` objects into shared memory using async
  copy-like machinery.
- Worker queues and scheduler queues are circular buffers in GPU memory.
- Task IDs encode iteration number and task index in 64 bits.
- Event IDs encode owner GPU and event index, with an NVSHMEM tag bit.
- The scheduler special-cases massive events, dependent-task events, end of graph
  events, and termination.

### Dynamic Serving Features

MPK includes LLM-serving-specific metadata inside the runtime:

- `tokens`, `input_tokens`, `output_tokens`
- `step`
- `qo_indptr_buffer`
- paged KV metadata
- request ids and page queues

The scheduler processes the end-of-graph event by preparing the next batch:
retiring completed requests, admitting new requests, updating KV metadata, and
launching the next iteration. This moves serving-loop logic from CPU to GPU.

This is powerful, but it means MPK is not a thin general CUDA layer. It is a
large model-serving backend with task-specific assumptions.

### Performance Claims

From the paper:

- End-to-end serving improves by about 1.0x to 1.7x over optimized vLLM/SGLang
  setups depending on model/GPU/batch.
- Multi-GPU tensor parallelism improves by about 1.1x to 1.4x over optimized
  serving systems.
- Cross-task pipelining gives about 1.2x to 1.3x on the evaluated final linear
  layer.
- Compute/communication overlap reduces per-iteration latency by about 1.1x in
  their multi-GPU ablation.

### Strengths

- Real system, public code, and end-to-end LLM-serving focus.
- Deep CUDA integration, including generated task device code, TMA/CUTLASS/CuTe
  style kernels, NVSHMEM, and Blackwell/Hopper/Ampere specialization.
- The event/task representation is concrete enough to execute efficiently.
- The runtime puts scheduling on GPU, removing CPU launch and request scheduling
  bottlenecks for decode loops.

### Weaknesses or Design Costs

- It is not thin. The repo is a full compiler/runtime stack with Python,
  generated CUDA, task registration, model-specific operators, and serving
  metadata.
- The API is still high-level and model-backend oriented, not "write normal
  CUDA but make it a megakernel".
- Generic dynamic scheduling has unavoidable overhead: queues, atomics, polling,
  descriptor loads, and scheduler CTAs.
- Supporting arbitrary user CUDA kernels is hard because MPK task descriptors
  assume registered task variants and generated code.
- The runtime reserves SM resources for schedulers. This is a reasonable
  tradeoff for dynamic workloads, but it is not zero overhead for regular static
  pipelines.

## Event Tensor Notes

### Problem Framing

Event Tensor starts from the same launch/barrier bottleneck but focuses on
dynamism:

- Dynamic shapes from continuous batching and variable sequence lengths.
- Data-dependent computation from MoE routing.
- Different scheduling strategies needed for different workloads.
- Megakernels are difficult to program manually because fine-grained dependency
  graphs are large and error-prone.

The paper's key claim is that current megakernel approaches lack a clean
compiler abstraction for dynamic task dependencies.

### Core Abstraction: Event Tensor

An Event Tensor is a multidimensional tensor whose elements are events. Each
event element represents completion of one logical set of producer tasks and can
enable consumer tasks. Instead of materializing millions of graph edges, the
program uses coordinate maps:

```text
producer tile coordinate -> event tensor coordinate
event tensor coordinate -> consumer tile coordinate(s)
```

This is the intuitive jump: events are not a flat runtime list; they have the
same structure as the tiled dataflow. If the data is tiled by `(batch, head,
block)`, the events can also be shaped by symbolic `(B, H, block)` dimensions.

### Shape Dynamism

Event Tensor uses symbolic shapes to describe a family of dependency graphs.
For example, an Event Tensor can have shape `(B, tiles)`, where `B` is the
runtime batch size. The compiler can AOT compile the dependency logic once, then
instantiate concrete event extents at runtime without recapturing CUDA Graphs or
JIT-compiling per shape.

This is valuable for serving warmup. The paper reports Qwen3-32B warmup of:

- SGLang JIT: 583 seconds and 51 graph captures.
- vLLM JIT: 123 seconds and 67 graph captures.
- Event Tensor AOT: 35 seconds runtime initialization, with no runtime graph
  capture. The offline compile is reported separately.

### Data-Dependent Dynamism

MoE is the motivating example. Routing decisions decide which expert gets which
tokens. The event dependency is not fully known at compile time.

Event Tensor handles this with:

- Data-dependent event update: runtime tensors like `topk` decide which producer
  tasks notify which expert event.
- Data-dependent task triggering: runtime prefix arrays like `exp_indptr` decide
  how many GroupGEMM tiles each expert event triggers.

The important insight is that the dependency chain can remain feed-forward even
though the specific fan-in/fan-out is runtime-dependent. TopK is computed first,
then later event updates/triggers use that data.

### Scheduling Transformations

Event Tensor supports both static and dynamic scheduling as compiler
transformations.

Static scheduling:

- Host/compiler computes per-SM queues.
- Persistent kernel loops through assigned tasks.
- Event Tensor lowers to integer counters.
- Producers call `notify`; consumers call `wait`.
- Best for predictable work because it avoids scheduler queue overhead.

Dynamic scheduling:

- Event completion pushes ready tasks into a GPU scheduler queue.
- Idle SMs pop tasks and execute them.
- Best for irregular or data-dependent workloads.
- Costs more due to queue atomics and runtime scheduling.

The paper explicitly presents this as a tradeoff rather than one universal
runtime.

### Lowering to Minimal Runtime

A strong point: Event Tensor aims not to require a heavy task-graph runtime.
After compilation, most scheduling logic is inlined into the generated
megakernel. Event tensors lower to integer tensors/counters; the dynamic
scheduler needs a task queue, but the full graph is not materialized as a
runtime object.

This is highly aligned with Megacu's zero-overhead goal.

### Performance Claims

From the paper:

- GEMM + ReduceScatter and AllGather + GEMM on 8 B200s reach up to 1.40x over
  cuBLAS+NCCL baseline.
- MoE layer reaches up to 1.23x over specialized baselines at 1024 tokens.
- Qwen3-30B-A3B serving reaches 1.48x over vLLM and 1.20x over SGLang at batch
  size 1.
- Qwen3-32B single-GPU serving reaches up to 1.15x over vLLM at batch size 1.
- TP=4 Qwen3-32B is roughly competitive with vLLM, but can trail SGLang in some
  settings due to serving-engine overhead and less-tuned generated GEMMs.

### Strengths

- Best abstraction among the two papers for shape and data-dependent dynamism.
- Makes dependency structure compact and compiler-friendly.
- Cleanly separates static and dynamic scheduling choices.
- AOT story is compelling for production warmup.
- The abstraction is DSL-agnostic in principle: TVM in their implementation, but
  they state it can fit Triton or CuteDSL-like stacks.

### Weaknesses or Design Costs

- No public implementation found, so details cannot be audited like MPK.
- The abstraction is compiler-centric. Expert CUDA programmers may not want to
  move into TVM or another DSL just to write a megakernel.
- Dynamic scheduling is not free; centralized queues can contend at scale.
- Static scheduling can handle dynamism only conservatively or by sampling
  representative shapes.
- "AOT dynamic shape" still needs careful bounds, buffer planning, and maximum
  shape assumptions.

## Comparison

| Axis | MPK | Event Tensor | Megacu Opportunity |
| --- | --- | --- | --- |
| Primary artifact | Public compiler/runtime repo | Paper and arXiv source | New device-native library |
| Main abstraction | SM task/event graph | Tensor-shaped event IR | Platform-native task/event layer |
| User level | PyTorch/Python model backend | Compiler IR/TVM DSL | Expert native-kernel API |
| Scheduler | In-kernel workers/schedulers, hybrid AOT/JIT | Static and dynamic compiler transforms | Static-first, dynamic opt-in |
| Dynamism | Serving-specific runtime logic, graph variants | Symbolic event tensors plus data-dependent event maps | Bounded symbolic shapes, explicit runtime maps |
| Runtime cost | Worker queues, scheduler queues, atomics, descriptor loads | Minimal for static, queue overhead for dynamic | Zero-cost static wrappers, explicit-cost dynamic runtime |
| Sell point | Automatic LLM megakernel backend | Dynamic megakernel compiler abstraction | "Megakernels as close to the native platform as possible" |

## Proposed Megacu Design

### Product Positioning

Megacu should not try to beat MPK or Event Tensor by being a bigger compiler.
The original CUDA-first position was:

> Megacu is a zero-overhead C++/CUDA layer for building persistent megakernels.
> Users write normal CUDA/CuTe/CUTLASS/NVSHMEM device code. Megacu only supplies
> typed task descriptors, event tensors, static schedules, and optional GPU-side
> scheduling primitives.

The updated position keeps the zero-overhead and expert-user thesis, but changes
the product boundary: Megacu is a platform-neutral device-native composition
layer, with CUDA/NVSHMEM as the first platform/backend pair and performance
baseline. The target user is an expert native-kernel programmer who does not
want Python graph capture, a heavy compiler IR, or hidden scheduling decisions.

### Design Principles

1. CUDA first.
   Device tasks are ordinary `__device__` or inlined C++ callables. They can use
   raw CUDA, PTX, CuTe, CUTLASS, TMA, cp.async, warp specialization, cooperative
   groups, and NVSHMEM directly.

2. Static schedule first.
   The default path should compile to a single persistent CUDA kernel with
   precomputed per-SM queues and direct `wait/notify` counters. No scheduler CTA
   is present unless requested.

3. Pay only for explicit features.
   If a graph uses no dynamic dispatch, there is no dynamic queue. If a task has
   no dependency, there is no wait. If the graph is known at compile time, task
   dispatch should be a switch or template dispatch that nvcc can inline.

4. Host API is graph construction, not performance abstraction.
   The host side builds task and event metadata. The hot path remains CUDA.

5. Event tensors are views, not a runtime graph.
   Represent event tensors as typed counter buffers plus coordinate mapping
   functions. Do not materialize edge lists unless the user selects a debug or
   dynamic mode that needs them.

6. Explicitly expose costs.
   The API should make atomics, waits, task queues, scheduler CTAs, and remote
   signals visible concepts. That makes "zero overhead" credible.

### Minimum API Shape

The first version can be header-only plus a small optional runtime library.

Sketch:

```cpp
using namespace megacu;

struct GemmTile {
  __device__ void operator()(task_ctx& ctx, Params const& p) const {
    // User writes normal CUDA/CuTe/CUTLASS code here.
  }
};

auto g = graph_builder{}
  .tensor("x", x_ptr, shape<B, H>{})
  .tensor("w", w_ptr, shape<H, O>{})
  .event_tensor("ready", shape<B, OTile>{})
  .task<GemmTile>("gemm", grid(B, OTile), block<128>())
    .reads(x, tile_map::rows_cols(...))
    .reads(w, tile_map::cols(...))
    .writes(y, tile_map::rows_cols(...))
    .notify(ready, map::same<0, 1>())
  .task<NormTile>("norm", grid(B, OTile), block<128>())
    .wait(ready, map::same<0, 1>())
    .reads(y)
    .writes(z)
  .schedule(static_round_robin{})
  .build();

launch_persistent(g, stream);
```

This is illustrative, not a final API. The important part is the shape:
CUDA task code remains user-owned; Megacu declares dependencies and launch
policy around it.

### Core Components

1. `task<T>`
   A compile-time task type wrapping a user device callable. It stores only what
   the callable needs: pointers, strides, scalar params, task coordinates, and
   optional metadata.

2. `event_tensor<Rank>`
   A typed view over an integer counter buffer. It supports `notify(coord)` and
   `wait(coord)` in device code. Shape can be static, bounded symbolic, or
   runtime-provided within a max extent.

3. `tile_map`
   A coordinate mapping layer. This is the thin equivalent of Event Tensor's
   lambda/einsum maps. It defines which event coordinate a task coordinate
   updates or waits on.

4. `static_schedule`
   Host-generated per-SM queues. Best for regular pipelines and compute/comm
   overlap. Queue entries should be compact and type-specialized.

5. `dynamic_queue`
   Optional GPU queue for irregular workloads. This should be a separate policy,
   not the default runtime. Candidate implementations: centralized MPMC queue
   first, then per-SM/cluster queues with work stealing if contention matters.

6. `persistent_kernel`
   A generated or templated CUDA kernel with one CTA per worker SM by default.
   Optional scheduler CTAs only when using dynamic scheduling.

7. `debug_graph`
   Optional JSON/DOT dump compatible in spirit with MPK's task graph tooling.
   This should never be part of the optimized path.

### Runtime Modes

Mode 1: `static_inline`

- Per-SM schedule is known before launch.
- Worker CTA loops over its queue.
- Dependency waits are direct counter waits.
- Task calls are templated/inlined when possible.
- No scheduler CTA.
- Best claim to true zero overhead.

Mode 2: `static_descriptor`

- Per-SM schedule still known before launch.
- Queue entries contain compact descriptors.
- One switch dispatches among task types.
- Useful when graph is built at runtime but should still avoid GPU scheduling.

Mode 3: `dynamic_local`

- Ready events push tasks into GPU queues.
- Workers pop tasks.
- Good for MoE, data-dependent routing, variable attention work.
- Explicitly not zero overhead; it is controlled overhead.

Mode 4: `hybrid`

- Static for regular regions.
- Dynamic only around irregular subgraphs.
- Barriers or event groups return execution to static schedule.
- This mirrors MPK/Event Tensor tradeoffs but keeps the choice local.

### How Megacu Differs From MPK

- MPK hides much of the system behind a PyTorch/Python backend; Megacu should be
  a C++/CUDA authoring layer.
- MPK has a fixed set of registered LLM task variants; Megacu should let users
  provide arbitrary device callables.
- MPK's runtime is dynamic by default in spirit; Megacu should make static
  scheduling the zero-overhead default.
- MPK's serving runtime is valuable but domain-heavy; Megacu should keep serving
  utilities outside the core.

### How Megacu Borrows From MPK

- Use `TaskDesc`/`EventDesc` style compact runtime metadata.
- Encode task ids and event ids in 64 bits.
- Keep event fan-out as contiguous ranges where possible.
- Prefetch task descriptors into shared memory in descriptor mode.
- Keep profiling hooks and graph dumps.
- Support NVSHMEM event signaling as an opt-in extension.

### How Megacu Borrows From Event Tensor

- Make event tensors first-class, not ad hoc scalar semaphores.
- Support coordinate maps from task coordinates to event coordinates.
- Support symbolic/bounded shapes for dynamic batch sizes.
- Keep static and dynamic scheduling as separate transformations/policies.
- Lower events to integer counters and inline notify/wait logic.
- Avoid runtime graph materialization in the optimized path.

### Real Zero-Overhead Definition

Megacu should be precise about "zero overhead":

- Zero abstraction overhead does not mean synchronization is free.
- It means the abstraction emits the same kind of CUDA operations an expert
  would write by hand.
- In `static_inline` mode, wrappers should compile away after inlining.
- Event operations are explicit atomics/spin-waits. They are algorithmic costs,
  not library overhead.
- Dynamic scheduling is never marketed as zero overhead. It is an opt-in feature
  for workloads where load balance wins more than queues cost.

### First Milestone

Build a minimal C++/CUDA prototype with:

1. One persistent kernel launcher.
2. One worker CTA per SM.
3. Static per-SM queues.
4. `event_tensor<1>` and `event_tensor<2>` counter buffers.
5. Two user-defined CUDA tasks with direct `wait/notify`.
6. A GEMM-like producer plus reduction/elementwise consumer microbenchmark.
7. A baseline handwritten persistent kernel to prove no measurable wrapper
   overhead.

Do not start with full LLM serving. The first correctness/performance target
should be a small graph where expected overlap is easy to inspect in Nsight.

### Second Milestone

Add:

- `static_descriptor` mode with compact descriptors and task switch dispatch.
- Graph dump and timeline profiling.
- Event fan-out contiguous range encoding.
- Shape-bounded event tensors.
- One compute/communication overlap demo using NVSHMEM or CUDA multimem if
  available.

### Third Milestone

Add:

- Dynamic queue policy.
- Data-dependent event update/trigger primitives.
- MoE-style toy benchmark.
- Hybrid static/dynamic scheduling.

### Design Risks

- If the API becomes a compiler DSL, Megacu loses its unique position.
- If dynamic scheduling becomes default, the zero-overhead claim becomes weak.
- If task descriptors are too generic, descriptor load and switch overhead will
  dominate small tasks.
- If symbolic shapes are too ambitious early, buffer planning will delay the
  core proof.
- If no baseline handwritten kernel is maintained, performance claims will be
  hard to trust.

### Recommended Direction

Use MPK and Event Tensor as conceptual references, but make Megacu deliberately
lower-level:

```text
Event Tensor idea + MPK runtime lessons + CUDA-native API + static-first policy
```

The product should feel like a small CUDA extension library, not a model
compiler. If an expert can look at the generated kernel and say "I would have
written basically this by hand", the design is on track.
