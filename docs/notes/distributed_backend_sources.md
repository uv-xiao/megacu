# Distributed Backend Source Notes

- Date: 2026-04-22 Asia/Shanghai
- Purpose: inform Megacu's backend interface, multi-GPU model, and project
  organization before the first device-native design is accepted.
- Related design: `docs/design/megacu_cpp_cuda_layer.md`

## Follow-up Design Correction

After the initial reading, the user clarified that Megacu should not be tightly
bound to CUDA. The source-reading lesson is therefore stronger than "thin
CUDA layer": Megacu should keep a platform-neutral task/event/schedule core,
then provide CUDA as the first platform adapter and NVSHMEM/MSCCL++/P2P as
transport backends. Triton-distributed's semantic vocabulary and MSCCL++'s
channel/device-handle layering both support this split.

On 2026-04-23, the user corrected the additional source to
`https://arxiv.org/pdf/2604.19241v1`. I downloaded it as
`research/papers/arxiv_2604_19241v1.pdf` and extracted text to
`research/sources/arxiv_2604_19241v1.txt`.

## Sources Read

### Triton-distributed

- Repository: `research/repos/triton-distributed`
- Upstream: https://github.com/ByteDance-Seed/Triton-distributed
- Local revision: `bec05d7`, branch `main`
- Submodules: `3rdparty/triton` and `3rdparty/mori` are present in
  `.gitmodules` but not initialized for this reading.
- Files read:
  - `README.md`
  - `docs/primitives.md`
  - `python/triton_dist/language/distributed_ops.py`
  - `python/triton_dist/language/extra/libshmem_device.py`
  - `include/TritonDistributed/Dialect/Distributed/IR/DistributedOps.td`
  - `python/triton_dist/mega_triton_kernel/core/{task_base.py,graph.py,scheduler.py,builder.py,code_generator.py}`
  - `python/triton_dist/mega_triton_kernel/kernels/{task_context.py,allreduce.py}`
  - `python/triton_dist/mega_triton_kernel/tasks/allreduce.py`
  - `python/triton_dist/kernels/nvidia/allgather_gemm.py`
  - 2026-04-24 follow-up for implementation-ready examples:
    `python/triton_dist/kernels/nvidia/gemm_allreduce.py`,
    `python/triton_dist/kernels/nvidia/allgather_gemm.py`,
    `python/triton_dist/mega_triton_kernel/kernels/allreduce.py`

### HazyResearch Megakernels / MegaKittens

- Repository: `research/repos/hazy-megakernels-mk-v2-llama-70b`
- Upstream: https://github.com/HazyResearch/Megakernels/tree/mk-v2-llama-70b
- Local revision: `c919cd0`, branch `mk-v2-llama-70b`
- Submodule initialized:
  - `csrc/ThunderKittens` at `839deba681677f581806ee81dd74d5fc764f2baf`
- Files read:
  - `README.md`
  - `megakittens/{backend.py,interface.py,scheduler.py,dispatcher.py,multi_dispatcher.py,tracer.py}`
  - `megakittens/schema/{dag.py,device.py,dtype.py,instruction.py,itype.py,tensor.py}`
  - `csrc/{schema.cuh,controller.cuh,workers.cuh,megakittens.cuh}`
  - `csrc/itypes/gemm.cuh`
  - targeted search over `csrc/itypes/reference`, `megakittens/itypes/llama70b`,
    and ThunderKittens parallel kernels for `pgl`, `dev_idx`, barriers, peer
    access, and all-device synchronization.

### MSCCL++

- Repository: `research/repos/mscclpp`
- Upstream: https://github.com/microsoft/mscclpp
- Local revision: `eeea00b`, branch `main`
- Files read:
  - `README.md`
  - `docs/overview.md`
  - `docs/programming_guide.rst`
  - `docs/tutorials/{01-basic-concepts.md,03-memory-channel.md,04-port-channel.md}`
  - `docs/guide/memory-management.md`
  - `docs/dsl/concepts.md`
  - `include/mscclpp/{core.hpp,memory_channel.hpp,memory_channel_device.hpp,port_channel.hpp,port_channel_device.hpp,semaphore_device.hpp,fifo_device.hpp,copy_device.hpp,device.hpp}`

### UniEP

- Paper: `UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training`
- arXiv: https://arxiv.org/abs/2604.19241
- Local PDF: `research/papers/arxiv_2604_19241v1.pdf`
- Local text: `research/sources/arxiv_2604_19241v1.txt`
- Date in paper: April 22, 2026.

## What Triton-distributed Contributes

Triton-distributed is useful because it separates a small distributed
programming vocabulary from backend-specific lowering. The public language layer
contains rank queries, symmetric pointer lookup, notification, waiting, and a
token consume operation. The lower layer dispatches SHMEM-like device APIs to
NVSHMEM, ROCSHMEM, Mori, or Metax variants through a proxy module.

The most relevant distributed primitives are:

- `rank(axis)` and `num_ranks(axis)`: expose topology identity in the kernel.
- `symm_at(ptr, rank)`: converts a symmetric local pointer into a peer pointer.
- `notify(ptr, rank, signal, sig_op, comm_scope)`: emits a remote signal with
  operation and communication scope.
- `wait(barrierPtrs, numBarriers, scope, semantic, waitValue)`: creates an
  acquire-style wait over one or more barriers.
- `consume_token(value, token)`: forces a data dependency from a wait into a
  later use.
- SHMEM device calls: `putmem`, `getmem`, `signal`, `barrier`, `sync`, `fence`,
  `quiet`, block/warp/thread variants, remote pointer calls, and multimem
  accessors.

The useful design lesson is not "copy Triton syntax." It is that a narrow,
backend-independent semantic layer can be lowered to multiple transports if the
semantic layer names memory ordering, communication scope, and peer identity.
Megacu should define those concepts directly in C++ rather than hide them under
an opaque runtime API.

The mega-kernel side of Triton-distributed uses:

- `TaskBase` records `task_type`, `layer_id`, `task_id`, `tile_id_or_start`,
  `num_tiles`, dependencies, IO tensor encodings, and extra params.
- `Graph` infers tensor producer/consumer dependencies from tensor storage.
- `scheduler.py` maps tasks to per-SM work queues and builds a scoreboard.
- The generated kernel fetches per-SM tasks, waits on scoreboard dependency
  ranges, dispatches the task body by task type, then releases tile readiness.
- Runtime scheduling exists as an optional mode using an atomic work queue.

That structure is close to Megacu's intended direction, but the encoding is
Python/Triton-specific. The lesson for Megacu is to keep a compact task
descriptor with predictable layout, preserve a static per-worker schedule as
the default, and make dynamic queues an explicit optional scheduling mode.

The Triton-distributed allreduce examples also matter because they use
NVSHMEM/multimem-specific operations inside task kernels. Megacu should allow
optimized task implementations to specialize for backend capabilities without
polluting the generic task graph. A generic task can declare "requires remote
multimem reduce" or "requires symmetric pointer + signal"; the concrete backend
then supplies the exact device operations.

Follow-up reading on 2026-04-24 made this concrete enough for the active design
example. `kernels/nvidia/gemm_allreduce.py` defines a fused shape with:

- symmetric GEMM output buffers and AllReduce output buffers;
- GEMM barrier buffers, tile barrier buffers, grid barrier buffers, and
  multi-store barrier buffers;
- communication CTAs waiting for GEMM tile readiness;
- GEMM CTAs computing output tiles and setting per-tile readiness;
- a fused kernel that assigns low program ids to communication work and the
  remaining program ids to GEMM work;
- backend-specific paths for multimem load-reduce/store and load-reduce-store
  fallback.

That is the right first Megacu validation example because it exercises all
important boundaries at once: logical output-tile domains, rank/team
participants, event readiness, dispatcher placement of compute versus
communication work, backend primitive selection, and native kernel bodies. It
also strengthens the "no generated CUDA" requirement: Megacu should link a
handwritten or library-provided GEMM implementation and backend reduction
primitive rather than synthesize a new CUDA kernel from the orchestrate
descriptor.

Follow-up reading on 2026-04-25 focused on how Triton-distributed implements
the fused GEMM+AllReduce kernel:

- `GemmARContext` owns symmetric GEMM output, symmetric AllReduce output, GEMM
  barrier buffers, tile barrier buffers, grid barrier buffers, multi-store
  barrier buffers, the communication stream, the number of communication SMs,
  and the selected AllReduce method.
- `kernel_fused_gemm_allreduce` launches one cooperative grid. Program ids
  below `NUM_COMM_SMS` run communication work; the remaining program ids run
  long-running GEMM work. The important Megacu lesson is that compute and
  communication roles can be represented as explicit participants without
  making the dispatcher own scheduler policy.
- `kernel_persistent_gemm_notify` is the compute side. Each GEMM CTA loops over
  output tiles assigned by `tile_id += NUM_GEMM_SMS`, computes one tile, stores
  the tile into a symmetric GEMM output buffer, then releases readiness through
  `gemm_barrier_ptr`. The barrier granularity is selected by `TILE_MAP_LEVEL`:
  tile-wise, row-wise, or rank-wise.
- `consumer_all_reduce_kernel` and
  `consumer_all_reduce_load_store_kernel` are the communication side. Each
  communication CTA loops over assigned output tiles or chunks, waits until all
  peer GEMM barriers for that tile/chunk are ready, then reduces peer symmetric
  GEMM outputs into the final AllReduce output. One path uses multimem
  load-reduce/store; the fallback uses explicit peer loads through symmetric
  pointers.
- `multi_st_barrier_ptr` coordinates the multimem store path after
  communication CTAs publish reduced values to symmetric output.
- `barrier_on_this_grid(grid_barrier_ptr, use_cooperative)` is used after the
  fused compute/communication sections so the cooperative grid has an
  intra-kernel completion point before optional output copy.
- `low_latency_gemm_allreduce_op` double-buffers the context phase, resets
  barriers when needed, and launches the fused kernel with grid size
  `NUM_COMM_SMS + min(NUM_GEMM_SMS, num_output_tiles)`.
- `gemm_allreduce_op` is the two-kernel stream variant. It launches the
  long-running GEMM notifier on the current stream, launches the consumer
  AllReduce on a separate high-priority communication stream, and then makes the
  current stream wait on the communication stream. Megacu should not rely on
  stream sequencing inside the scheduler design.

The Megacu example should mirror the important semantics, not the Triton syntax:
reserve logical compute and communication participants, use a tiled GEMM work
loop, signal tile readiness, have communication work wait on readiness before
reducing, and keep backend-specific reduction mechanics behind a narrow target
ops interface.

## What UniEP Contributes

UniEP is a focused mega-kernel system for expert-parallel MoE training. It is
important because it demonstrates that mega-kernel fusion is not only an
inference trick: for training, it can fuse Dispatch+GroupGEMM and
GroupGEMM+Combine to coordinate communication and computation while preserving
bitwise numerical consistency.

The paper's core techniques are:

- long-running worker threadblocks, one per physical SM;
- dynamic SM role assignment among communication, computation, relay, and
  reduction work;
- token-level deterministic global mapping so parallel dispatch preserves the
  same token order as sequential execution;
- global-memory scoreboard segments for token arrival and tile readiness;
- relay workers that trade local HBM copies for reduced NVLink traffic when
  several top-k experts live on the same destination rank;
- a parameterized search space over worker counts and warp allocation;
- an analytical performance model and cached autotuning to choose configurations
  without manual kernel switching.

The positioning lesson for Megacu is sharp. UniEP is a vertical system:
excellent for EP MoE training, tightly connected to token routing, GroupGEMM,
numerical reproducibility, and Triton-distributed implementation mechanics.
Megacu should not compete by building another MoE mega-kernel. It should expose
the lower-level reusable substrate that makes UniEP-like systems easier to
write and verify: typed device-visible events, explicit memory semantics,
platform/backend capability checks, role/schedule policies, and direct access
to NVSHMEM/MSCCL++/platform primitives.

UniEP also argues against an overly small "just events" API. Real systems need
to express:

- producer/consumer order at token or tile granularity;
- role assignment and participation policy;
- deterministic ordering constraints, especially for training;
- explicit resource partitioning among compute, communication, relay, and
  reduction work;
- backend-specific transport choices without changing the logical dependency
  model.

This does not mean Megacu should ship a heavy `core` or `memory` framework. It
means the core vocabulary must be just large enough to name those facts without
locking users into a domain-specific compiler.

## What MegaKittens Contributes

MegaKittens is useful as a model of project organization for composable
optimized instructions. A new operation is represented by:

- a Python `IType` metadata class that declares inputs, outputs, block indices,
  validation, tensor access regions, in-place behavior, tests, and C++ include
  path;
- a generated instruction stream containing compact fixed-size instruction
  records;
- a C++ instruction implementation that participates in a fixed worker model;
- a scheduler that computes tensor allocation, reuse, and fine-grained barriers.

The most important schema details:

- `instruction_t` is fixed at 256 bytes.
- Source tensor ids, destination tensor ids, instruction indices, source
  barriers, barrier targets, barrier counts, and destination barriers are all
  explicit fields.
- The scheduler builds barriers from per-instruction access regions rather than
  only coarse node dependencies.
- Tensor reuse is represented as a first-class dependency so memory reuse is
  not a hidden side effect.

The worker split is also important:

- `controller` fetches instructions and manages instruction pipeline slots.
- `loader`, `launcher`, `consumer`, and `storer` specialize work across warps.
- Instruction types expose nested roles (`controller`, `loader`, `launcher`,
  `consumer`, `storer`) so each optimized kernel can own its internal pipeline.
- The kernel itself is a normal CUDA kernel compiled through NVRTC/CUBIN and
  launched with explicit grid, block, dynamic shared memory, and cluster
  settings.

For Megacu, the lesson is that a "thin CUDA layer" should not mean one generic
kernel body that hides all optimization. It should mean a stable outer contract
around task metadata, event metadata, backend handles, and launch descriptors,
while task implementations remain free to use handwritten CUDA, TMA, WGMMA,
cluster barriers, multimem, inline PTX, and backend-specific fast paths.

The current MegaKittens multi-device path is informative but not sufficient as
a direct architecture:

- `Dispatcher` is primarily single-device and has explicit TODO comments for
  multi-GPU handling.
- `MultiDispatcher` enables peer access, materializes per-device tensors, packs
  PGL descriptors with all peer pointers, compiles one kernel, loads it into
  every device context, launches per device, then synchronizes devices.
- PGL-oriented kernels carry `NUM_DEVICES` and `dev_idx` into device code.
- ThunderKittens parallel kernels use PGLs for peer tensor access and explicit
  barrier tensors/signals across devices.

The Megacu implication is that multi-GPU must be part of the core descriptor
model: world/team identity, local rank, global rank, peer pointer tables,
remote event tensors, and per-device launch packaging should not be bolted on
after a single-device dispatcher exists.

## What ThunderKittens Contributes

ThunderKittens' parallel kernels show that optimized multi-GPU kernels often do
not want a high-level collective call. They want:

- a parallel tensor/global pointer layout with local and peer slices;
- a device-local rank (`dev_idx`);
- explicit barriers or signal arrays in peer-addressable memory;
- launch-time conversion from a Python/C++ object into a packed global
  descriptor;
- access to raw CUDA mechanisms such as TMA, cluster synchronization, and
  multimem-like paths.

The all-gather GEMM and ring-attention examples show compute and communication
interleaved inside the kernel. The backend layer therefore cannot be only a
library of full collectives. It must expose primitive remote operations and
event objects that handwritten kernels can compose at tile granularity.

## What MSCCL++ Contributes

MSCCL++ is the strongest source for backend layering. It has three levels:

- primitive C++/CUDA APIs;
- a Python DSL over primitives;
- an NCCL-compatible API over the DSL/runtime.

The primitive model is built from:

- `Bootstrap`: rank/world exchange and barriers for setup.
- `Endpoint` and `Connection`: asymmetric peer connectivity.
- `SemaphoreStub` and `Semaphore`: explicit signal/wait resources.
- `RegisteredMemory`: non-owning registered memory regions.
- `MemoryChannel`: direct peer memory access by GPU threads.
- `PortChannel`: GPU-triggered, host/proxy-serviced transfer through ports
  such as copy engines or RDMA.
- `ProxyService`: host-side service backing `PortChannel` FIFO requests.
- `DeviceHandle`: compact handles copied to or referenced by device code.

The device-side split is especially relevant:

- `MemoryChannelDeviceHandle` exposes remote `read`, `write`, collective-thread
  `put`, collective-thread `get`, packet put/unpack, `signal`, `wait`,
  relaxed signal/wait, and poll.
- `PortChannelDeviceHandle` exposes one-thread trigger operations:
  `put`, `putWithSignal`, `putWithSignalAndFlush`, `signal`, `flush`, `wait`,
  and `poll`.
- `SemaphoreDeviceHandle` distinguishes memory-ordering signal/wait from
  relaxed signal/wait.
- The port FIFO uses a fixed trigger format and system-scope stores/atomics.

This directly argues for a Megacu backend interface with capability classes,
not one universal "copy" function. A direct peer-memory backend and a
proxy/port backend have different participation rules:

- peer memory copy: many GPU threads may participate in the operation;
- port/proxy transfer: usually one GPU thread triggers work and a host proxy or
  hardware engine performs the transfer;
- symmetric-memory/NVSHMEM: device calls may expose peer pointers, remote
  puts/gets, barriers, quiet/fence, and team scopes;
- multimem/NVLS-style paths: use special allocation and addressing with
  different requirements from normal device memory.

MSCCL++ also shows ownership and lifetime boundaries that Megacu must name:

- registered memory does not own the underlying allocation;
- serialized endpoints, stubs, and registered memory metadata must remain alive
  while peers use them;
- host channel objects must outlive device handles used by kernels;
- channels may require host services that must be started before launch and
  stopped after all kernels complete.

## Comparison

| Source | Strongest feature | Design risk if copied directly | Megacu lesson |
| --- | --- | --- | --- |
| MPK | Megakernel scheduling and fusion direction | Compiler/runtime choices may be heavier than a thin CUDA layer | Preserve explicit task/event/schedule concepts; avoid hiding costs. |
| Event Tensor | Event-as-tensor readiness and fine-grained dependencies | Event representation can become too abstract if not mapped to CUDA memory/order | Model event tensors as explicit device data with ownership and memory semantics. |
| Triton-distributed | Small distributed primitive layer lowered to backend-specific SHMEM APIs | Python/Triton IR is not Megacu's C++ surface | Define a small C++ semantic layer: rank, peer ptr, signal, wait, fence/quiet, put/get. |
| MegaKittens | Modular instruction metadata plus fixed CUDA worker runtime | Current public path is mostly single-device; C++ pipeline is specialized | Use metadata traits and compact descriptors; keep task bodies handwritten CUDA. |
| ThunderKittens | PGL/parallel tensor patterns and optimized distributed kernels | Very hardware- and model-specific kernels | Support backend-specific fast paths without changing task graph concepts. |
| MSCCL++ | Clean channel/device-handle/resource lifetime model | Full runtime/DSL is broader than Megacu's zero-overhead goal | Separate control plane, device channel handles, capabilities, and host services. |

## Backend Interface Implications

Megacu should split backend support into four layers:

1. Host control plane
   - world/rank initialization;
   - team/subgroup construction;
   - endpoint/connection/bootstrap exchange where needed;
   - memory registration and symmetric allocation;
   - host proxy service lifecycle where needed;
   - per-device launch package construction.

2. Device transport handle
   - compact POD or trivially copyable handle;
   - callable from `__device__` code;
   - no virtual dispatch in device hot paths;
   - exposes only capabilities that the backend can implement efficiently.

3. Semantic operation vocabulary
   - local rank/global rank/team rank;
   - local pointer, symmetric pointer, remote pointer, registered region,
     peer tensor, and event tensor references;
   - `put`, `get`, `read`, `write`, `signal`, `wait`, `poll`, `fence`,
     `flush`, `quiet`, barrier, and optional packet operations;
   - memory ordering and communication scope are explicit parameters or
     type-level policies.

4. Scheduler/task integration
   - tasks declare platform and backend capability requirements;
   - event dependencies carry owner rank, scope, expected value, and memory
     semantics;
   - schedule construction fails early when the selected platform/backend pair
     lacks a required operation;
   - backend-specific task specializations are allowed under the same logical
     task id.

The device hot path should be template/concept based:

```cpp
template <class Platform, class Backend>
MEGACU_DEVICE(Platform)
void task(task_ctx<Platform, Backend> ctx) {
  auto peer = ctx.backend().peer(ctx.remote_rank());
  ctx.backend().put(peer, ctx.remote_dst(), ctx.local_src(), bytes,
                    thread_participation::block());
  ctx.backend().signal(ctx.remote_event(), memory_semantic::release);
}
```

That sketch should compile to direct backend calls after inlining. A runtime
polymorphic host `backend_runtime` can exist, but it must not leak into device
task dispatch.

## Proposed Capability Groups

Backend capability names should be positive and narrow:

- `rank_query`: device code can query rank/world/team identity.
- `symmetric_ptr`: local symmetric pointer can be mapped to peer pointer.
- `registered_region`: backend uses registered memory handles and byte offsets.
- `peer_pointer_table`: launch package carries raw peer pointer table.
- `remote_read_write`: scalar remote load/store are valid.
- `remote_put_get_thread`: one-thread put/get or trigger operation.
- `remote_put_get_block`: block/grid/thread-group copy operation.
- `signal_wait_relaxed`: execution-only signal/wait.
- `signal_wait_ordered`: signal/wait with memory visibility semantics.
- `fence_quiet_flush`: explicit completion/fence APIs.
- `packet_ll`: packetized small-message protocol.
- `multimem_reduce`: multimem/NVLS-style reduction or broadcast primitive.
- `host_proxy`: device operation enqueues requests for a host service.
- `team_scope`: operations can target teams/subgroups, not only global ranks.

The first NVSHMEM backend likely supports rank query, symmetric pointer,
put/get, signal/wait, fences/quiet, barriers, teams, and some multimem paths on
supported systems. An MSCCL++ memory-channel backend supports registered
regions, direct peer memory read/write/put/get, and signal/wait, but has
different allocation and handle requirements. An MSCCL++ port-channel backend
supports proxy-triggered put/signal/flush and must expose `host_proxy` so the
scheduler does not assign many GPU threads to a one-thread trigger path.

## Project Organization Implications

Megacu should keep generic concepts and backend implementations physically
separate:

```text
include/megacu/
  core/          # ids, ranks, teams, errors, spans, platform-neutral config
  memory/        # tensor_ref, peer_tensor, registered_region, symmetric_region
  event/         # event_ref, event_tensor, local/remote event ownership
  platform/      # platform concepts, launch contracts, device attrs
  platform/cuda/ # CUDA platform adapter, launch, streams, inspection
  task/          # task concept, task metadata, task_ctx, task registry traits
  schedule/      # static schedules, optional dynamic queues, descriptors
  kernel/        # long-running kernel templates and launch descriptors
  backend/       # backend concepts and common semantic wrappers
  backend/nvshmem/
  backend/mscclpp/
  backend/cuda_p2p/
src/
  platform/cuda/
  backend/nvshmem/   # host setup, allocation, registration, launch packing
  backend/mscclpp/
  backend/cuda_p2p/
examples/
  single_gpu/
  multi_gpu/
benchmarks/
tests/
  compile/
  unit/
  cuda_runtime/
docs/
```

The important rule is that `task/`, `event/`, and `schedule/` can depend on
platform and backend concepts, but they should not include concrete platform or
backend headers unless an example or test intentionally selects them. Concrete
platforms and backends adapt to Megacu contracts; Megacu contracts should not
become CUDA or NVSHMEM contracts with generic names.

## Candidate Backend Architecture Options

### Option A: NVSHMEM-first implementation, generic concepts around it

- Build core abstractions around NVSHMEM's actual device capabilities first.
- Keep MSCCL++ and CUDA P2P as documented future adapters.
- Fastest path to a running multi-GPU benchmark.
- Risk: public concepts may accidentally inherit NVSHMEM-specific assumptions
  such as symmetric allocation everywhere.

### Option B: Capability-first core with NVSHMEM as first adapter

- Define backend concepts and capability traits first.
- Implement only the NVSHMEM adapter initially.
- Keep examples constrained to operations NVSHMEM supports, but make the task
  API speak Megacu concepts such as event tensors, peer regions, and memory
  semantics.
- More upfront design work.
- Lower risk of forcing later MSCCL++ or P2P support through an NVSHMEM-shaped
  abstraction.

### Option C: Transport-neutral graph IR before any backend

- Design a backend-independent graph/IR and lower it to NVSHMEM/MSCCL++ later.
- Best if the project is mainly a compiler framework.
- Highest delay before CUDA-native proof points.
- Risk: violates the user's "close to CUDA" goal by pushing work into an
  abstract planning layer too early.

Recommendation: Option B. It matches the user's request: multi-GPU from the
beginning, backend support designed carefully, NVSHMEM first, and no hot-path
abstraction overhead. Option B lets Megacu implement a narrow executable slice
without making the public API NVSHMEM-only.

## Design Constraints To Promote

- Device backend calls must be statically resolved by templates/concepts in
  performance-sensitive code.
- Host-side backend selection may use runtime polymorphism or factories because
  it is outside the device hot path.
- Events are device data, not opaque host runtime promises.
- Every remote event must carry owner rank/team, address/offset, expected
  value, memory semantics, and scope.
- Every remote memory reference must say whether it is a symmetric pointer,
  registered region offset, peer pointer table entry, or backend-native handle.
- Dynamic scheduling requires explicit opt-in and accounting for queue atomics,
  polling, and scheduler occupancy.
- Backend-specific optimized kernels are allowed, but the logical task metadata
  and schedule/event contracts must remain backend-neutral.
- Proxy-backed transports must be declared as such so kernels do not accidentally
  treat them like direct peer-memory operations.

## Verification Evidence For This Note

- Repository revisions checked with `git -C <repo> rev-parse --short HEAD`.
- Hazy ThunderKittens submodule checked with `git -C ... submodule status`.
- Focused source reads used `sed`, `find`, and `rg` over the listed files.
- No performance claims are made here; source examples are treated as design
  evidence only.
