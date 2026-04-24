# Dispatcher, Scheduler, And Kernel Lowering

This chapter separates three things that were previously too entangled.

## Dispatcher

The dispatcher has two build-graph parts:

1. a reusable compiled implementation
2. an orchestrate-target lowering step on one concrete program

Together they map logical program work to execution-level placements.

Inputs from the program model:

- named op
- logical work domain
- virtual participant placement
- resource usage
- explicit event dependencies
- optional mapping hints

Dispatcher responsibilities:

- decide how logical work is partitioned
- decide placement groups or execution lanes
- produce the execution-level work units consumed by the scheduler

The dispatcher is independent from the scheduler. This matches the user's point:
mapping strategy and scheduling strategy are not the same thing.

Implementation owner: `src/dispatcher/`.

First dispatcher contract:

- input: lowered program facts from the public builder
- output: dispatch table mapping logical domain instances to execution
  placements
- output: participant table mapping virtual participants at each domain point
  to backend-native peers, ranks, lanes, or CTAs
- no ownership of event wait/signal semantics
- no ownership of execution order
- no public authoring API beyond optional placement hints

## Scheduler

The scheduler also has two build-graph parts:

1. a reusable compiled engine
2. an orchestrate-target payload specialized for one concrete program

The scheduler consumes dispatcher output. It does not define the author's public
API.

Examples:

- static persistent schedule
- queue-backed schedule
- scheduler-CTA model
- cluster-synchronous model

Implementation owner: `src/scheduler/`.

First scheduler contract:

- input: dispatcher output plus explicit event dependencies
- output: schedule payload for the selected execution strategy
- no public scheduler base class
- no runtime scheduler selection
- static persistent scheduling is the first implemented strategy

## Kernel Lowering

Kernel lowering also splits into reusable compiled logic plus orchestrate-target
specialization.

It decides how named kernels are turned into executable device behavior for the
chosen scheduler.

Examples:

- keep named kernels as separate launches inside the orchestrate-controlled
  execution path
- stitch several named kernels into one persistent kernel
- generate helper glue around backend/platform interaction

This is also where fine-grained overlap needs careful treatment.

Implementation owner: `src/lowering/`.

First kernel-lowering contract:

- input: named op table, typed resource table, event table, dispatch table, and
  selected scheduler payload
- input: participant table from dispatcher and backend event-layout rules
- output: CUDA/NVSHMEM executable path plus inspection metadata
- owns stitching and launch payload construction
- does not invent public fragment-op APIs

## Fine-Grained Compute/Communication Overlap

The user concern is correct: fine-grained overlap is important, and a model that
treats every op as a fully opaque launched kernel would be too weak.

The chosen solution is:

- keep a single public named-op abstraction
- allow backend primitives inside named kernel bodies
- let kernel lowering stitch named kernels when the chosen target strategy
  supports it

So fine-grained overlap comes from two places:

1. in-kernel backend/platform primitives used inside named kernels
2. reusable engines plus orchestrate-target stitching/composition by the
   lowering pipeline

Megacu does not need a public fragment taxonomy to support this.

## Build-Graph Ownership

Dispatcher, scheduler, and kernel lowering implementations are compiled first,
then chosen and specialized during orchestrate-target lowering.

Runtime orchestration must not choose among them.

## Backend Resolution Contract

Backend adapters own the final translation from Megacu slots to native backend
objects.

For CUDA/NVSHMEM first, `src/backends/nvshmem/` must provide:

- `nvshmem_team_view`: typed runtime handle passed into the orchestrate program;
- event-storage layout rules for `remote_event`;
- peer resolution from dispatcher participant slots to NVSHMEM PE ids;
- signal/wait helpers callable from lowered kernels;
- host/runtime validation that the supplied team and event storage match the
  target envelope.

Initial device-side API shape:

```cpp
namespace megacu::nvshmem {
struct event_endpoint;

__device__ void signal(
    megacu::cuda::kernel_context ctx,
    event_endpoint endpoint,
    std::uint64_t value);

__device__ void wait(
    megacu::cuda::kernel_context ctx,
    event_endpoint endpoint,
    std::uint64_t value);
}
```

`event_endpoint` is created by lowering and accessed through
`kernel_context::event<tag>(...)`. User kernels should not compute NVSHMEM
signal addresses directly unless they intentionally bypass Megacu for a
handwritten baseline.

## Kernel Context Contract

Lowered kernels receive a platform-specific context. For CUDA first:

```cpp
namespace megacu::cuda {
struct kernel_context {
  template <class DomainTag>
  __device__ domain_point<DomainTag> domain_point() const;

  template <class ParticipantTag, class DomainPoint>
  __device__ backend_peer peer(DomainPoint point) const;

  template <class EventTag, class DomainPoint>
  __device__ megacu::nvshmem::event_endpoint event(
      DomainPoint point,
      backend_peer peer) const;
};
}
```

The context is target-specific. It may be a compact pointer to generated
metadata, inline constants, or registers produced by lowering. Its observable
contract is typed lookup by tag, not string lookup by name.

## Internal Records

The orchestrate target lowering may create internal records, but those records
are not public authoring APIs.

Minimum internal records for the first slice:

- op table: named op symbol, implementation symbol, domain reference
- resource table: typed view slots used by lowered code
- event table: event storage, scope, release/acquire dependencies
- dispatch table: logical domain to execution placement
- participant table: virtual participant to backend peer/rank/lane mapping
- schedule payload: static persistent execution order
- kernel payload: named kernel entrypoints and stitched execution metadata
- backend payload: NVSHMEM handles and signal/wait metadata needed by kernels

Each record must have one owner:

- public program builders collect semantic facts;
- dispatcher owns placement;
- scheduler owns execution order;
- lowering owns kernel stitching and launch payloads;
- platform/backend adapters own native handles and primitive calls.

No record should duplicate a concept already owned by another layer.

## Implementation Architecture

The implementation should likely organize these as separate internal components:

- dispatcher implementation
- scheduler implementation
- kernel-lowering implementation
- target-lowering driver that applies them to one concrete orchestrate program
- CUDA platform adapter under `src/platform/cuda/`
- NVSHMEM backend adapter under `src/backends/nvshmem/`

But they are not separate public authoring layers. Their composition is part of
the CMake-managed build graph that produces one compiled orchestrate target.
