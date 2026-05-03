# Program, Compile, And Execute Flow

This file gives an intuitive end-to-end picture of how Megacu is used. It is
not a new API layer. It explains the intended flow for the CUDA+NVSHMEM first
target.

## One Sentence

The user writes a normal host-callable orchestrate function. Inside it, the user
submits small native kernel operators. CMake links that orchestrate function
with a selected dispatcher, scheduler, CUDA platform adapter, NVSHMEM backend,
and native operator kernels. At runtime, host code calls the orchestrate
function once; that one Megacu megakernel runs a common device-side
dispatcher/scheduler loop and invokes the linked operator kernels internally on
one or more GPUs.

## Mental Model

```text
small native operators
  gemm_tile_f32(...)
  phased_allreduce_f32(...)
        |
        | referenced by
        v
Megacu orchestrate function
  submit(gemm_tile_f32, raw args..., attrs(...))
  submit(phased_allreduce_f32, raw args..., attrs(depends_on(...)))
        |
        | compiled and linked by CMake with a ConfigureTarget
        v
one Megacu target operation
  gemm_allreduce_phased_orchestrate(driver, a, b, partial, out, m, n, k)
        |
        | called by host/framework code
        v
one Megacu megakernel execution
  dispatcher maps local/peer work
  scheduler orders explicit dependencies
  platform/backend launch and communicate through CUDA+NVSHMEM resources
  linked operator kernels run internally
```

The host sees one target call. The target may run multiple linked operator
kernels inside one Megacu device-side running loop. That is the Megacu meaning
of "one megakernel" in this design.

## What The User Programs

The user does not write a program IR, materializer input, target metadata, or a
runtime graph. The user writes ordinary C++ orchestration around native
operator calls.

For the tiny phased GEMM+AllReduce example:

```cpp
extern "C" megacu::status gemm_allreduce_phased_orchestrate(
    megacu::cuda_nvshmem::driver &driver,
    const float *a,
    const float *b,
    float *partial,
    float *out,
    int m,
    int n,
    int k) {
  megacu::scope phase(driver);

  auto ready_events = phase.event_tensor(attrs(
      events::shape{m_tiles, n_tiles},
      cuda_nvshmem::events::symmetric_i32_storage{events},
      cuda_nvshmem::events::team_scope{}));

  auto gemm = phase.submit(
      gemm_tile_f32,
      driver, a, b, partial, problem,
      attrs(
          dispatcher::tile_grid{m_tiles, n_tiles},
          events::publish(ready_events)));

  auto ready = phase.sync(
      attrs(
          scheduler::depends_on(gemm),
          events::join(ready_events)));

  phase.submit(
      phased_allreduce_f32,
      driver, partial, out, problem,
      attrs(
          scheduler::depends_on(ready),
          events::acquire(ready_events)));

  return phase.finish();
}
```

The important parts are:

- `driver` is the execution handle supplied by host code.
- `a`, `b`, `partial`, `out`, `m`, `n`, and `k` are normal target arguments,
  shaped like CUDA kernel arguments.
- `gemm_tile_f32` and `phased_allreduce_f32` are linked native operator
  implementations, not generated code.
- `phase.submit(...)` passes raw operator arguments directly.
- `phase.sync(...)` declares a sync-only task with no operator body.
- `attrs(...)` carries the explicit facts Megacu components need.
- `scheduler::depends_on(...)` is the dependency edge. Megacu does not infer
  edges from pointer use.
- `events::publish/join/acquire(...)` are event tensor handlers attached to
  tasks. They lower inside the selected platform/backend implementation.

The `submit` parsing rule is intentionally thin: the linked operator signature
determines how many raw arguments are consumed. An optional trailing `attr_set`
attaches component attributes. A sync-only task plus an orchestration-owned
event tensor is Megacu's native form of the Event Tensor middle node between
producer and consumer tasks; the scheduler/backend lower the handlers to
internal counters, waits, signals, or ready-queue pushes.

## What The Driver Is

The driver is not a target-argument bag. It is the host execution object for the
selected platform/backend.

For CUDA+NVSHMEM, the driver includes:

- CUDA stream and device/context identity, because host code uses them to
  enqueue and synchronize asynchronous GPU work;
- NVSHMEM PE, team identity, and team size, because the backend needs them to
  know local and peer communication identity;
- launch/process facts, because single-process, `nvshmrun`, MPI, or future
  framework launch modes must all be normalized before the target runs.

The driver excludes:

- target tensors and scalars;
- operator argument packs;
- target-local scratch and event storage;
- linked dispatcher/scheduler/backend/platform component sets;
- target capability envelopes.

Those are either normal target arguments or linked target implementation facts.
Keeping them out of the driver prevents the driver from becoming a generic
runtime environment.

## Program Flow

The Megacu "program" is not an IR. It is the C++ orchestrate body plus linked
native operators and attributes.

For GEMM+AllReduce:

```text
GEMM task
  operator: gemm_tile_f32
  raw args: a, b, partial
  attributes:
    dispatcher::tile_grid{m, n, k}
    events::publish(ready_events)

Sync task
  operator: none
  attributes:
    scheduler::depends_on(GEMM)
    events::join(ready_events)

AllReduce task
  operator: phased_allreduce_f32
  raw args: partial, out
  attributes:
    scheduler::depends_on(Sync)
    events::acquire(ready_events)
```

There is no hidden dependency lookup. If the AllReduce must wait for GEMM, the
orchestrator must say so explicitly. If the author omits that dependency,
Megacu does not scan pointers and add it back.

## Compilation Flow

CMake does normal compile and link work:

```text
compile Megacu runtime components
  dispatcher: spatial mapping
  scheduler: dependency/progress policy
  platform: CUDA launch/resource facts
  backend: NVSHMEM team/device primitive support
  target runtime: common status and component sequencing

compile user target code
  gemm_allreduce_phased_orchestrate.cc

compile native operator kernels
  gemm_tile_f32
  phased_allreduce_f32

link one OrchTarget
  orchestrate entry
  selected ConfigureTarget components
  native operator symbols
  small capability facts
```

Compilation does not:

- translate an orchestrate program into another language;
- build a program IR;
- run a materializer;
- generate CUDA source;
- create static dispatch or schedule sections.

After linking, the result is just a native C++/CUDA target exposing a direct
entry point such as:

```cpp
gemm_allreduce_phased_orchestrate(driver, a, b, partial, out, m, n, k)
```

## Host Execution Flow

Host or framework code owns the usual asynchronous execution surface. For
CUDA+NVSHMEM, each participating rank builds a local driver and calls the same
target entry.

```text
host rank 0 / PE 0                         host rank 1 / PE 1
  creates CUDA stream                         creates CUDA stream
  selects local device                        selects local device
  joins NVSHMEM team                          joins NVSHMEM team
  prepares symmetric storage                  prepares symmetric storage
  builds cuda_nvshmem::driver                 builds cuda_nvshmem::driver
  calls orchestrate(...)                      calls orchestrate(...)
```

The target arguments stay the same for one GPU or two GPUs. The driver tells
the linked backend and dispatcher what execution world the local rank is in.

## Runtime Inside The Megakernel

Once host code calls the target, Megacu runs the linked runtime path:

```text
1. Target runtime sequences the linked components for this call.

2. The orchestrate body submits native operators with raw arguments and
   explicit attributes, and may submit sync-only tasks for readiness
   transforms.

3. Dispatcher maps submitted work:
     1-device: local work only
     2-device: local work plus peer communication work

4. Scheduler consumes only explicit dependency attributes:
     GEMM -> sync-only readiness task -> AllReduce

5. CUDA platform adapter exposes simple launch facts.

6. NVSHMEM backend exposes team facts and device-side communication
   primitives.

7. The Megacu megakernel dispatches and schedules linked operator kernels:
     gemm_tile_f32
     phased_allreduce_f32

8. Status returns to host code.
```

From the host viewpoint, this is one Megacu operation. Inside the operation,
the dispatcher and scheduler can run several operator kernels in the required
order without exposing per-operator launches or stream synchronization to user
code.

## Distributed CUDA+NVSHMEM Flow

CUDA supplies the local GPU execution mechanism. NVSHMEM supplies the
distributed device-side communication mechanism.

For one host with two GPUs:

```text
PE 0 driver:
  CUDA stream for GPU 0
  NVSHMEM team size 2
  PE identity 0
  symmetric storage/session facts

PE 1 driver:
  CUDA stream for GPU 1
  NVSHMEM team size 2
  PE identity 1
  symmetric storage/session facts
```

Both PEs call:

```cpp
gemm_allreduce_phased_orchestrate(driver, a, b, partial, out, m, n, k);
```

The dispatcher uses the driver-backed backend identity to decide local and peer
work. The all-reduce operator then uses linked NVSHMEM device primitives
internally.

The target author does not write a separate single-card program and multi-card
program. The same target call retargets through the linked dispatcher and
backend resources.

## Why This Is Thin

Megacu stays thin because it avoids extra public layers:

- no program IR;
- no materializer;
- no generated source;
- no generic runtime argument frame;
- no task `input`/`output`/`inout` wrappers;
- no dependency inference from tensors or pointers;
- no runtime registry that chooses platform/backend/scheduler by string.

The only author-facing ideas are:

- direct target arguments after a driver;
- raw operator arguments in `submit`;
- explicit component attributes;
- linked dispatcher, scheduler, platform, backend, and native operator code.

## What Can Go Wrong

Megacu intentionally does not hide these responsibilities:

- If an operator needs ordering, the target author must attach the dependency
  attribute.
- If a communication buffer must be symmetric, the target author or adapter
  must provide the required backend attribute/resource contract.
- If a target supports only a dtype, layout, team size, or shape range, that
  belongs in the linked capability facts and validation path.

The design chooses explicitness over fallback inference so the runtime remains
small, inspectable, and close to what an expert CUDA+NVSHMEM programmer would
write by hand.
