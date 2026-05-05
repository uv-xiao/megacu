# Design: General Runtime-Linked Components

## Goal

Generalize the PR #4 runtime-linked implementation while keeping Megacu thin.
This PR must support required MPI and Torch launch adapters, multiple linked
dispatcher/scheduler/runtime strategies, and three end-to-end examples:
GEMM-RS, AG-GEMM, and a tiny decode pipeline. It must not become a graph
runtime, compiler/materializer, or heavy validation layer.

## Context

PR #4 proved the architecture with one CUDA+NVSHMEM GEMM+AllReduce target.
That target remains useful history, but it is not the example scope for this
PR. The current implementation must move to reusable tile-operator examples.
Several implementation details are still too narrow:

- attributes are encoded as one central enum/union;
- only one host-side dispatcher and scheduler path exists;
- only one device-runtime loop shape exists;
- distributed launch is limited to direct local/Docker execution;
- MPI and Torch are documented as future work rather than executable adapters.
- there is no set of end-to-end examples proving the same runtime components
  across distributed producer/consumer communication and non-GEMM pipelines.

The next PR changes that boundary. MPI and Torch are required, not optional.
Docker is the required execution environment for those dependencies.
Additional example/design evidence comes from Triton-distributed and
HazyResearch Megakernels, recorded in
`docs/notes/general_runtime_examples_sources.md`.

## Alternatives

1. **Optional adapters, default build stays dependency-light.**
   Rejected. It lets MPI/Torch silently fall out of the tested PR and does not
   meet the requested architecture completeness bar.
2. **Required Docker environment, adapters built and tested in Docker.**
   Selected. Local builds may remain useful, but Docker is the source of truth
   for the full PR because it owns CUDA, NVSHMEM, MPI, PyTorch, and launch
   tooling versions.
3. **Full framework integration with production PyTorch extension packaging.**
   Rejected for this PR. The required Torch path should prove
   `torchrun`/`torch.distributed` launch normalization into the Megacu driver,
   not package a full Python extension stack.

For the Hazy-style end-to-end example:

1. **Full Llama1B/HuggingFace decode.**
   Rejected for this PR. It would require external model downloads, large model
   memory, and framework packaging that would dominate the runtime-linked
   architecture work.
2. **Tiny decode pipeline modeled after Hazy's instruction sequence.**
   Selected. It exercises multiple operator kinds, explicit readiness, schedule
   strategy selection, and golden/baseline/Megacu comparison while staying
   deterministic in Docker.
3. **Only a single MLP benchmark.**
   Rejected as too small. Hazy's value is the end-to-end decode pipeline and
   instruction/barrier scheduling shape, not only a stack of matmuls.

For distributed examples:

1. **Keep GEMM+AllReduce as the main example.**
   Rejected for this PR. It proved the previous reset boundary, but the current
   PR needs examples that exercise reusable distributed task composition rather
   than one older proof shape.
2. **Implement GEMM-RS and AG-GEMM from tile operators.**
   Selected. GEMM-RS follows Triton-distributed tutorial 08's producer GEMM
   tiles feeding reduce-scatter work. AG-GEMM follows tutorial 07's all-gather
   data movement feeding GEMM tiles. In Megacu these must be written as
   operator tasks, sync-only tasks, explicit dependencies, dispatcher attrs, and
   EventTensor attrs.
3. **Write each example as one handwritten fused/persistent kernel.**
   Rejected for Megacu examples. A handwritten mega-kernel can be kept only as
   a baseline when useful. The Megacu variant must compose small tile operators
   into one mega-kernel through runtime, scheduler, dispatcher, and EventTensor
   components.

## Selected Design

### Required Launch Adapters

Add first-class launch adapters that normalize launch environments into the
same CUDA+NVSHMEM driver used by the orchestrate function:

```text
direct local process
  -> cuda_nvshmem::driver

MPI-launched process
  -> MPI rank/local-rank facts
  -> NVSHMEM MPI bootstrap or equivalent Docker-supported path
  -> cuda_nvshmem::driver

torchrun process
  -> torch.distributed rank/local-rank facts
  -> NVSHMEM UID/bootstrap exchange through torch control plane
  -> cuda_nvshmem::driver
```

Adapters own launch normalization and prelaunch environment checks. They do not
choose task placement, dependency order, or scheduler policy. After the driver
is built, the same orchestrate function runs.

### Strategy Selection

Make dispatcher, scheduler, runtime execution-model, and runtime-loop choices
explicit linked components:

- Dispatcher strategies:
  - `tile_grid`: local tile iteration;
  - `rank_aware_tile_grid`: same submitted task shape retargeted by backend PE
    identity;
  - `single_work_item`: non-GEMM contract-test dispatcher.
- Scheduler strategies:
  - `explicit_asap`: runs ready tasks as soon as dependencies allow;
  - `static_submission_order`: deterministic explicit-dependency scheduler
    for predictable static work;
  - `static_level_order`: groups tasks by explicit dependency depth so tests
    can verify strategy selection without relying on one task order.
- Runtime execution models:
  - `host-orch`: host orch builds and seals a complete task arena before
    launch;
  - `seeded-orch`: host seeds launch facts and device orch publishes concrete
    task records inside the mega-kernel.
- Runtime loop strategies:
  - `block_tile_runtime`: block/tile mega-kernel loop;
  - `single_work_item_runtime`: minimal reusable test runtime;
  - `grid_stride_runtime`: general runtime for fixed-size independent work
    ranges;
  - CUDA launch wrapper strategy selected by target configuration.

Strategy selection happens in target/configuration code and CMake-linked
components. Normal target authors still write orchestrate functions and
operators; they do not program `scheduler.run`.

The host-called orchestrate function launches one mega-kernel through the
selected runtime. The CUDA stream belongs to host-side enqueue and final
synchronization only. Inside the mega-kernel, the runtime execution model owns
task publication and sealing, while the runtime loop combines scheduler
readiness, dispatcher availability, and EventTensor progress. Internal
readiness is lowered by the selected EventTensor component and its
platform/backend implementation; it is not expressed as CUDA stream waits
between small kernels.

### Naming Decision: Runtime, Not Entry

The component formerly called `entry` is renamed to `runtime`. This is the
right name because the component owns the runtime execution model and
device-side mega-kernel loop, not just a launch symbol.

To avoid ambiguity:

- `orchestrate function` means the host-called target ABI;
- `runtime execution model` means the runtime-owned task construction,
  publication, and sealing policy;
- `runtime loop` means the linked device-side loop policy;
- `target call glue` means common host-side sequencing/status code that was
  previously called target runtime in older notes.

The old component name `entry` is rejected for this design because it hides the
most important ownership point: different runtimes may use different execution
models and loops while sharing the same scheduler, dispatcher, EventTensor,
platform, and backend components.

### Orchestration Model

Each orchestrate call owns one orchestration frame:

```text
orchestration_frame
  raw target args
  operator bindings
  tasks[]
  event_tensors[]
  linked component config
```

Tasks and event tensors are separate call-owned objects. Both use the same attr
mechanism, but their attrs mean different things:

```cpp
struct task_record {
  task_ref id;
  operator_ref op;    // present for submit, empty for sync
  raw_arg_view args;  // operator task only
  attr_set attrs;
};

struct event_tensor_record {
  event_ref id;
  attr_set attrs;
};
```

`submit(...)` injects `task::operator_task{}` into the task attrs. `sync(...)`
injects `task::sync_task{}` into the task attrs. Users do not write task-kind
attrs directly in this PR, and custom task kinds are out of scope.

Event tensor object attrs describe the event object:

```text
event_tensor::shape{...}
event_tensor::wait_count{...}
event_tensor::scope{...}
cuda_nvshmem::event_tensor::symmetric_storage{...}
```

Task attrs describe operations on event tensors:

```text
event_tensor::notify(e)
event_tensor::wait(e)     // sync-only task completion condition
event_tensor::trigger(e)  // dynamic scheduling candidate
```

Join, split, and barrier-like behavior are represented as sync-only tasks that
carry combinations of EventTensor operation attrs. They are not separate
operator kernels and not manually programmed waits/signals inside producer or
consumer operators.

The EventTensor component owns event tensor attrs, event operation attrs,
sync-task completion conditions, and backend/platform lowering. Runtime loops
call the EventTensor component. Schedulers own explicit task dependency order
and must not know CUDA/NVSHMEM event details. Dispatchers may use task kind and
dispatcher-owned attrs to choose work cursors, but they must not infer task kind
from `op == null` or from raw arguments.

The interaction is:

```text
runtime loop:
  for tasks in its chosen loop discipline:
    ask scheduler whether explicit task deps are ready
    for sync-only tasks, ask EventTensor whether wait conditions complete
    ask dispatcher whether this worker has a work cursor for this task
    invoke operator body or sync-task event handler
    report event progress and task progress
```

This makes Event Tensor native without putting event semantics inside the
runtime loop. Runtime owns the execution model and loop; EventTensor owns event
semantics.

The operation names intentionally follow the Event Tensor paper's
counter-shaped abstraction:

- an event tensor element starts from an explicit wait count;
- producer-side task attrs lower to `notify` on event tensors;
- sync-only task attrs lower to `wait` completion conditions;
- dynamic triggering is a future/strategy-specific extension, not required for
  the first static CUDA+NVSHMEM path.

These names describe the lowered semantics. Users still express them as attrs
attached by `submit(...)` or `sync(...)`; operator kernels do not call them.

### EventTensor API And Decoupling

EventTensor has three layers:

```text
common EventTensor API
  event_tensor_ref
  event_tensor::shape
  event_tensor::wait_count
  event_tensor::scope
  event_tensor::notify / wait / trigger attrs

platform EventTensor binding
  CUDA memory model
  block/thread visibility
  atomic/wait primitive choices
  device-side counter layout constraints

backend EventTensor binding
  NVSHMEM team/rank facts
  symmetric storage
  remote signal/wait/fence/quiet semantics
  peer-visible counter layout
```

The common API is the user-facing semantic layer. It says that an operator task
notifies an EventTensor when it completes, or that a sync-only task waits on an
EventTensor before completing. It does not name CUDA atomics, NVSHMEM signals,
symmetric allocation, or memory ordering primitives unless a platform/backend
attr is explicitly attached.

`scheduler::depends_on(...)` and `event_tensor::wait(...)` must coexist because
they have different roles. `depends_on` specifies task-record ordering.
`event_tensor::wait` specifies the condition for a sync-only task to complete.
EventTensor is not a second scheduler and is not a finer-grained dependency
system inside an operator task.

The platform/backend bindings provide concrete lowering. CUDA chooses local
counter representation, memory scope, and device-side wait/atomic mechanics.
NVSHMEM adds team/rank identity, symmetric storage requirements, remote
visibility, and communication ordering. The target configuration composes these
bindings:

```text
ConfigureTarget<
  runtime::device_persistent<
    runtime::execution::seeded_orch,
    runtime::loop::block_tile>,
  dispatcher::rank_aware_tile_grid,
  scheduler::explicit_asap,
  event_tensor::counter_tensor,
  platform::cuda,
  backend::nvshmem>
```

The API shape for target authors is:

```cpp
auto ready = orch.event_tensor(
    event_tensor::shape<2>{tiles_m, tiles_n},
    event_tensor::wait_count{driver.world_size()},
    event_tensor::scope::team{},
    cuda_nvshmem::event_tensor::symmetric_storage{});

orch.submit<gemm_tile_op>(
    gemm_args...,
    event_tensor::notify(ready));

auto ready_sync = orch.sync(
    scheduler::depends_on(gemm),
    event_tensor::wait(ready));

orch.submit<allreduce_op>(
    ar_args...,
    scheduler::depends_on(ready_sync));
```

The API shape for sync-only tasks is:

```cpp
orch.sync(
    event_tensor::wait(local_done),
    event_tensor::notify(global_ready));
```

The runtime loop sees EventTensor as a linked service:

```cpp
runtime.run(ctx, scheduler, dispatcher, event_tensor, operators);
```

Inside the mega-kernel loop, the runtime asks EventTensor to evaluate and lower
operation attrs around task issue:

```text
if (!scheduler.ready(task, ctx)) continue;
if (!dispatcher.cursor(task, worker, ctx)) continue;

if (task is sync-only)
  event_tensor.wait(task.attrs, ctx);
else
  invoke operator body;

event_tensor.notify(task.attrs, cursor, ctx);
scheduler.complete(task, cursor, ctx);
```

This keeps the component thin. EventTensor owns the semantic contract and hook
names; platform/backend own the native mechanisms. Users pick semantic attrs
and target configs, not per-task platform/backend sync calls.

Megacu intentionally differs from Event Tensor here: it does not use
EventTensor to split one operator task into finer subtasks. The operator task is
the sync granularity. If a target needs smaller-grained sync, it should submit a
smaller operator/task and connect it through normal task dependencies and
sync-only EventTensor tasks.

### EventTensor Versus The Paper

Megacu's EventTensor should not be described as less expressive than the
original Event Tensor paper. The better distinction is level of abstraction.

The paper is higher-level and compiler-assisted. It can describe a tiled
producer-to-event-to-consumer graph compactly with symbolic shapes and mappings,
then lower that graph into a generated mega-kernel. Megacu is lower-level and
explicit. The target author, a helper library, or a future generator must submit
the operator tasks, sync-only tasks, EventTensor objects, and `depends_on`
relations directly.

That means the same dependency shape can be represented in Megacu if the target
submits tasks at the desired granularity:

```cpp
auto ready = orch.event_tensor(
    event_tensor::shape{tile_count},
    event_tensor::wait_count{1});

auto produce = orch.submit<producer_tile_op>(
    tile_args,
    event_tensor::notify(ready));

auto ready_sync = orch.sync(
    scheduler::depends_on(produce),
    event_tensor::wait(ready));

orch.submit<consumer_tile_op>(
    tile_args,
    scheduler::depends_on(ready_sync));
```

The original Event Tensor compiler can generate this pattern from a compact
tile graph. Megacu exposes the lower-level task construction surface. A future
Megacu generator could produce the verbose submissions from a higher-level
Event Tensor-like description without changing the runtime-linked component
model.

The tradeoff is therefore:

- not expressiveness loss;
- more explicit authoring or generator burden;
- no automatic dependency inference or task splitting;
- lowering quality determines whether sync-only tasks compile down to
  comparable native waits/signals and task-state updates;
- performance depends on whether the target author chooses the right operator
  task granularity.

The design requirement is to keep this distinction precise. EventTensor is not
a second scheduler and not a hidden compiler graph, but it should still be able
to encode Event Tensor-style designs when those designs are written as explicit
Megacu tasks and sync-only EventTensor tasks.

### Attribute Shape

Move toward typed component-owned attributes. The public path should support
component-owned structs and typed lookup. A compatibility shim may keep the PR
small, but the new tests must prove non-GEMM attributes do not require
GEMM-specific enum slots or string lookup.

### Docker As Source Of Truth

The Docker image for this PR must include:

- CUDA toolkit/runtime compatible with local GPU execution;
- NVSHMEM;
- MPI runtime and development headers;
- Python and PyTorch with distributed support;
- scripts that run direct, MPI, and torchrun correctness paths.

Local non-Docker builds are allowed as a convenience, but they are not the full
verification story.

### Triton-Distributed Design Inputs

The PR should use Triton-distributed as a concrete source for:

- torchrun process-group launch and NVSHMEM unique-id bootstrap patterns;
- GEMM-RS and AG-GEMM distributed tile-producer/tile-consumer shapes;
- rank/local-rank/world-size normalization before backend initialization;
- keeping distributed primitive semantics narrow: rank, team size, symmetric
  pointer/storage, signal, wait, fence/quiet, and backend-specific reduction
  capability.

Megacu should not copy Triton syntax or Python/Triton graph construction. The
source-derived requirement is that Megacu's C++ runtime components expose the
same necessary execution facts without reintroducing a compiler/materializer.

### Required Example Shape

The PR should implement these three examples:

```text
examples/cuda_nvshmem/gemm_reduce_scatter/
  common/
  golden/
  baseline/
  megacu/

examples/cuda_nvshmem/allgather_gemm/
  common/
  golden/
  baseline/
  megacu/

examples/cuda_nvshmem/tiny_decode_pipeline/
  common/
  golden/
  baseline/
  megacu/
```

The Megacu variants must not encode each algorithm as one large manual
mega-kernel. They must expose the intended authoring model:

```text
orchestrate(driver, raw args...)
  -> declare EventTensor objects
  -> submit communication or compute tile operator tasks
  -> submit sync-only tasks where EventTensor wait conditions complete
  -> submit dependent consumer operator tasks
  -> launch one runtime-owned mega-kernel
```

For GEMM-RS:

```text
GEMM tile producer tasks
  -> EventTensor notify attrs
  -> sync-only readiness tasks with event_tensor::wait
  -> reduce-scatter tile/region consumer tasks
```

For AG-GEMM:

```text
all-gather segment/tile producer tasks
  -> EventTensor notify attrs
  -> sync-only readiness tasks with event_tensor::wait
  -> GEMM tile consumer tasks
```

For tiny decode:

```text
norm/projection tile or range tasks
  -> activation/residual tasks
  -> MLP-style tasks
  -> logits/head tasks
```

The operator task should be the synchronization granularity. If an example
needs finer readiness, it should submit smaller tile/range operator tasks, not
split one submitted task internally through EventTensor.

### Hazy-Style End-To-End Example

Add `examples/cuda_nvshmem/tiny_decode_pipeline/` with:

```text
common/     shared problem, buffers, and reference descriptors
golden/     straightforward expected-output implementation
baseline/   handwritten CUDA/native baseline
megacu/     Megacu orchestrate + linked operator tasks + strategy selection
```

The tiny decode pipeline should model the Hazy Llama decode shape at small
fixed dimensions:

```text
input hidden
  -> norm/qkv-like projection
  -> activation/residual stage
  -> MLP/up/down-like stage
  -> logits/head stage
```

It should use deterministic inputs, no external model downloads, and no large
framework dependency beyond the Docker environment already required for
Torch-launch verification. The purpose is correctness and architecture
coverage, not Llama performance.

Golden, baseline, and Megacu roles are separate:

- golden computes expected output simply;
- baseline is native CUDA or straightforward platform code without Megacu;
- Megacu uses the same authoring model as other targets: orchestrate submits
  raw operator tasks with explicit attrs, and linked components run the
  megakernel.

## Examples

- Example: Direct GEMM-RS
  - Feature shown: existing direct/local adapter builds the same driver and
    runs distributed-style tile producer/consumer work through Megacu.
  - Verification mapping: Docker direct correctness test.

- Example: MPI GEMM-RS
  - Feature shown: MPI launch adapter produces driver resources without
    changing the orchestrate function.
  - Verification mapping: Docker MPI-launched two-rank correctness test.

- Example: Torch GEMM-RS
  - Feature shown: torchrun/torch.distributed control plane produces driver
    resources without changing the orchestrate function.
  - Verification mapping: Docker torchrun two-rank correctness test.

- Example: AG-GEMM
  - Feature shown: communication producer tasks and GEMM consumer tile tasks
    use the same scheduler, dispatcher, runtime, and EventTensor ownership as
    GEMM-RS.
  - Verification mapping: Docker direct and distributed correctness tests.

- Example: Tiny decode pipeline
  - Feature shown: Hazy-style end-to-end operator sequence using reusable
    dispatcher/scheduler/runtime strategies.
  - Verification mapping: Docker correctness test comparing golden, baseline,
    and Megacu paths.

## Contracts

- MPI and Torch adapter targets are required in Docker and must not be skipped
  by missing-dependency fallbacks inside the Docker verification path.
- Triton-distributed and Hazy source-derived requirements must be recorded in
  `docs/notes/` before implementation claims.
- Launch adapters may validate launch environment facts, CUDA device selection,
  and backend bootstrap consistency.
- Launch adapters must not choose dispatcher placement or scheduler ordering.
- Dispatcher strategies own spatial work mapping only.
- Scheduler strategies own temporal readiness only.
- Runtime strategies own the device running loop and issue policy only.
- Operators keep raw signatures.
- Dependencies remain explicit attributes.
- Missing dependency attributes remain target-author responsibility; Megacu
  does not infer or repair them.

## Dispatcher/Scheduler/Runtime Candidate Design

Implementation must start by designing and reviewing candidates before adding
code. The candidates are selected only if their contracts are simple enough to
verify in this PR.

### Common Pattern

The common runtime shape is:

```text
orchestrate(driver, raw target args...)
  -> declare event tensors with explicit attrs
  -> submit operator tasks and sync-only tasks with explicit attrs
  -> target config links dispatcher + scheduler + runtime + EventTensor + platform + backend
  -> runtime launches one mega-kernel
  -> runtime-owned mega-kernel loop:
       query scheduler for logical readiness
       query EventTensor for event issue/progress
       query dispatcher for spatial work cursor availability
       invoke the linked operator body or sync-only event handler
```

The common device-side contract is intentionally small:

- scheduler output: logical readiness/progress state for a task;
- EventTensor output: event issue/progress/completion status for a task;
- dispatcher output: a compact work cursor for a task and worker context;
- runtime input: scheduler, dispatcher, driver/device resources, raw call frame;
- runtime output: status/error flag visible after the mega-kernel completes.

The scheduler never owns spatial placement. The dispatcher never owns temporal
readiness. The EventTensor component never owns the runtime loop. The runtime
loop never invents dependencies or spatial placement, but it does own the loop
discipline: scan order, spin behavior, buffering of ready-but-unissued tasks,
retry policy, and completion observation. The runtime execution model owns task
publication, sealing, and host/device orch placement.

Normal users do not select these pieces by calling component APIs inside the
orchestrate body. A target links a named configuration, for example:

```text
ConfigureTarget<
  runtime::device_persistent<
    runtime::execution::seeded_orch,
    runtime::loop::block_tile>,
  dispatcher::rank_aware_tile_grid,
  scheduler::explicit_asap,
  event_tensor::counter_tensor,
  platform::cuda,
  backend::nvshmem>
```

The exact C++ spelling can change during implementation, but the ownership
boundary cannot. Component selection is target/configuration code, not user
algorithm code.

### Selection Rules

- Pick `explicit_asap` whenever the example is meant to demonstrate internal
  readiness and Event Tensor-style sync-only tasks.
- Pick `static_submission_order` when deterministic replay is more important
  than earliest-ready progress.
- Pick `static_level_order` only for tests or examples that need to prove a
  second static scheduler without introducing a queue.
- Pick `rank_aware_tile_grid` for GEMM-RS and AG-GEMM because the same
  orchestrate code must retarget one-device and two-device runs.
- Pick `tile_grid` for local tiled kernels that do not need peer-aware cursors.
- Pick `single_work_item` for non-GEMM contract tests and tiny single-item
  stages; do not use it as the distributed proof.
- Pick `block_tile_runtime` for tiled CUDA work, `grid_stride_runtime` for
  independent ranges, and `single_work_item_runtime` for minimal contract
  tests.

Unsupported combinations fail during target configuration or prelaunch adapter
setup. They do not fall back to another strategy.

### Dispatcher Candidates

1. **`tile_grid`**
   - Input: tile-grid attrs plus backend team facts.
   - Output: block-local tile cursors.
   - Best fit: GEMM-RS, AG-GEMM, and any 2D tiled operator target.
   - Accepted for PR: yes, as the local tiled baseline dispatcher.
   - Evidence: unit/contract test that one target can map a 2D tile range
     without referring to GEMM-specific names.
   - Risk: can become GEMM-specific if it reads problem names instead of typed
     tile attrs.

2. **`rank_aware_tile_grid`**
   - Input: tile-grid attrs plus rank/team facts from the driver.
   - Output: local tile cursors and peer-aware communication cursors.
   - Best fit: 1-host/1-GPU and 1-host/2-GPU retargeting with the same
     orchestrate code.
   - Accepted for PR: yes, as the distributed tiled dispatcher.
   - Evidence: Docker direct one-rank and two-rank GEMM-RS and AG-GEMM tests
     using the same Megacu orchestrate code.
   - Risk: dispatcher must not choose task order or backend primitive details.

3. **`single_work_item`**
   - Input: an explicit count or a single default work item.
   - Output: one or more independent work ids.
   - Best fit: non-GEMM contract tests and the tiny decode pipeline.
   - Accepted for PR: yes, but only as a reuse/contract dispatcher.
   - Evidence: non-GEMM test target and tiny decode stage that compile without
     GEMM-specific attrs.
   - Risk: too small to prove distributed placement, so it complements rather
     than replaces tiled tests.

### Scheduler Candidates

1. **`explicit_asap`**
   - Input: explicit dependency attrs and dispatch state.
   - Output: logical ready status as soon as dependencies are satisfied.
   - Best fit: GEMM-RS, AG-GEMM, and Event Tensor-style sync-only tasks.
   - Accepted for PR: yes, as the primary scheduler.
   - Evidence: test where consumer tasks become runnable from explicit event
     attrs, without operator code performing waits or signals manually.
   - Risk: implementation must not infer missing dependencies.

2. **`static_submission_order`**
   - Input: submitted task list and explicit dependency attrs.
   - Output: deterministic readiness scan/order that respects dependencies.
   - Best fit: tiny decode pipeline and tests that need stable ordering.
   - Accepted for PR: yes, as the simplest non-ASAP scheduler.
   - Evidence: tiny decode or contract target that runs with the same submitted
     tasks under this scheduler.
   - Risk: should reject cycles or impossible dependencies, not silently skip.

3. **`static_level_order`**
   - Input: explicit dependency attrs.
   - Output: readiness grouped by dependency depth, with submission order as
     the tie-breaker within a level.
   - Best fit: proving multiple scheduler strategies without adding queues.
   - Accepted for PR: yes, if its construction stays bounded and testable.
   - Evidence: compile/runtime contract test showing a different order than
     submission order while preserving explicit dependencies.
   - Risk: level construction must stay compact and bounded.

Rejected scheduler candidates for this PR:

- stream-based scheduler: wrong abstraction because the orchestrate call
  becomes one mega-kernel; CUDA streams only enqueue and synchronize that outer
  launch;
- tensor-access inference scheduler: wrong abstraction because dependencies
  are explicit attrs, not inferred from pointer, tensor, input, output, or
  inout declarations;
- user-programmed `scheduler.run`: wrong API because scheduler readiness is
  Megacu internal component behavior.

### Runtime Candidates

1. **`block_tile_runtime`**
   - Runs dispatcher work items across CUDA blocks.
   - Best fit: tiled distributed examples such as GEMM-RS and AG-GEMM.
   - Accepted for PR: yes, as the CUDA tiled mega-kernel runtime.
   - Evidence: GEMM-RS and AG-GEMM Megacu paths use this runtime with linked
     scheduler and dispatcher components.

2. **`grid_stride_runtime`**
   - Runs a fixed-size work range with grid-stride loops.
   - Best fit: vector/tiny decode stages and reusable CUDA tests.
   - Accepted for PR: yes, as the general independent-range runtime.
   - Evidence: tiny decode stage or non-GEMM test target uses this runtime
     without changing operator signatures.

3. **`single_work_item_runtime`**
   - Runs exactly one logical work item.
   - Best fit: host/device contract tests and simple non-GEMM targets.
   - Accepted for PR: yes, as a minimal runtime for smoke and ABI tests.
   - Evidence: build/runtime contract test uses the same scheduler/dispatcher
     API with one work item.

Rejected runtime candidates for this PR:

- problem-specific GEMM/communication runtime: wrong boundary because task order and
  operator sequence must come from submitted tasks plus scheduler readiness;
- per-operator CUDA kernel launch: wrong boundary because Megacu's orchestrate
  call launches one mega-kernel, not separate host-dispatched kernels.

## Architecture Comparison

This comparison is source-derived from:

- `research/repos/hazy-megakernels-mk-v2-llama-70b/examples/llama1b/scheduler.py`
  and `csrc/schema.cuh`;
- `research/repos/simpler/docs/orchestrator.md`,
  `docs/scheduler.md`, and `docs/task-flow.md`;
- Triton-distributed tutorial 07, tutorial 08, and
  `python/little_kernel/design/test_flashcomm_torchrun.py`.

### Compared Shape

| System | Program object | Dependency source | Running loop owner | Communication/launch lesson | Megacu decision |
| --- | --- | --- | --- | --- | --- |
| MegaKittens/Hazy | Flat instruction list plus instruction metadata and barriers. | Explicit barrier ids in instruction records. | Device controller/worker runtime tied to instruction format and worker roles. | Good model for tiny decode pipeline and explicit barrier-style readiness. | Keep explicit task/event objects, but avoid adopting a fixed instruction ABI or model-specific worker taxonomy. |
| Simpler/PTO Runtime | DAG slots from orchestrator submissions with `Callable`, `TaskArgs`, `CallConfig`. | TensorMap lookup from input/output/inout tags. | Host scheduler thread with wiring, ready, and completion queues. | Good contrast for task handles and queue mechanics. | Reject tensor-access inference and host queue scheduling; Megacu uses explicit attrs and one device mega-kernel runtime. |
| Triton-distributed | Handwritten distributed kernels plus Torch/NVSHMEM launch context. | Kernel-specific barrier buffers and rank/tile protocols. | Kernel body or Python launch code owns the concrete algorithm. | Strong source for torchrun, rank/local-rank, NVSHMEM bootstrap, symmetric storage, and GEMM-RS/AG-GEMM tests. | Put launch facts in adapters/driver and event lowering in the EventTensor backend/platform component; do not copy Triton syntax or kernel-specific protocols into Megacu core. |

### Why Megacu Is Different

Megacu should sit between Simpler's host DAG runtime and the highly specialized
kernel systems:

- Like MegaKittens, Megacu accepts explicit event/barrier-style readiness and
  runs a mega-kernel-shaped path.
- Unlike MegaKittens, Megacu should not require a fixed instruction record,
  fixed worker roles, or model-specific tensor slots.
- Like Simpler, Megacu has an orchestrate layer that records tasks.
- Unlike Simpler, Megacu does not infer dependencies from tensor access tags and
  does not dispatch tasks through a host scheduler thread.
- Like Triton-distributed, Megacu treats distributed launch, rank facts,
  symmetric storage, and device-side communication as first-class execution
  facts.
- Unlike Triton-distributed, Megacu should expose reusable task/event,
  dispatcher, scheduler, runtime, platform, and backend components instead of
  embedding all scheduling and communication protocol choices inside one
  hand-written kernel.

This makes the renamed runtime component important. It is the place where
Megacu deliberately resembles mega-kernel systems: one device-side loop owns
issue policy. The loop remains replaceable so Megacu can test spin, buffered,
static scan, and range runtimes without changing task/event records or operator
signatures.

## Failure Modes

- Missing MPI/PyTorch in Docker is a hard build or test failure.
- Rank/world-size mismatch between MPI/Torch and NVSHMEM is a hard adapter
  error before launch.
- Unsupported scheduler/runtime/dispatcher combinations return `unsupported`
  with a small status detail; they must not silently fall back to another
  strategy.

## Verification

- `git diff --check`
- local configure/build/test for dependency-light feedback
- Docker configure/build/test with MPI and PyTorch present
- Docker direct GEMM-RS correctness
- Docker MPI GEMM-RS correctness
- Docker torchrun GEMM-RS correctness
- Docker direct/distributed AG-GEMM correctness
- Docker tiny decode pipeline golden/baseline/Megacu correctness
- strategy-selection contract tests
- non-GEMM reuse contract test

## Out Of Scope

- Full PyTorch C++ extension packaging.
- Full Hazy Llama1B/HuggingFace model download or production decode benchmark.
- Multi-node performance claims.
- Arbitrary PE-count performance tuning.
- Dynamic scheduler queues.
- Compiler/materializer or generated metadata revival.

## Closeout

Before merge, recreate the flat stable docs under `docs/design/` according to
`docs/in_progress/design/stable_docs_recreation_plan.md`, remove the old
stable PR #4 folder, update README entry points if needed, remove completed
active docs, and update `docs/todo/` so future work reflects what remains.
