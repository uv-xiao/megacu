# EventTensor Design

Megacu EventTensor is an orchestration object inspired by Event Tensor, but the
synchronization level is the Megacu task. If a target needs finer readiness, it
submits smaller operator tasks, such as one task per tile. Megacu does not split
one submitted task into hidden subtasks.

## Roles

`depends_on` and EventTensor waits coexist because they answer different
questions:

- `scheduler::depends_on(task)` and `scheduler::depends_on_many(group)` say
  when a task is schedulable according to explicit task dependencies.
- `event_tensor::wait(event, tile)` and
  `event_tensor::wait_strided(event, first_tile, count, stride)` say when a
  sync-only task has completed its event condition after the scheduler has
  allowed that sync task to run.

A common producer/sync/consumer pattern is:

```cpp
auto ready = orch.event_tensor(attrs(
    event_tensor::shape(tile_count, 1),
    event_tensor::wait_count(team_size),
    cuda_nvshmem::event_tensor::symmetric_storage(events),
    cuda_nvshmem::event_tensor::scope::team{}));

for (int64_t tile = 0; tile < tile_count; ++tile) {
  auto produced = orch.submit(op_slot::produce_tile,
      attrs(dispatcher::single_tile(tile), event_tensor::notify(ready)));
  auto ready_task = orch.sync(attrs(
      scheduler::depends_on(produced),
      event_tensor::wait(ready, tile)));
  orch.submit(op_slot::consume_tile,
      attrs(scheduler::depends_on(ready_task),
            dispatcher::single_tile(tile)));
}
```

The operator bodies only compute tile work. The runtime loop invokes the
EventTensor component after producer work and while completing sync-only tasks.

For fan-in patterns such as AG-GEMM over many K tiles, dependency groups and
strided EventTensor waits are separate attrs:

```cpp
auto deps = orch.dependency_group();
for (int64_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
  auto tile = k_tile * n_tiles + n_tile;
  auto produced = orch.submit(op_slot::gather_tile,
      attrs(dispatcher::single_tile(tile), event_tensor::notify(gathered)));
  deps.push(produced);
}

auto ready = orch.sync(attrs(
    scheduler::depends_on_many(deps.ref()),
    event_tensor::wait_strided(gathered, n_tile, k_tiles, n_tiles)));
```

The dependency group is scheduler-owned readiness state. The strided wait is
EventTensor-owned sync completion state. They are not substitutes for each
other.

## Implemented Lowering

The common attrs are platform neutral:

- EventTensor object attrs: `shape`, `wait_count`, storage, and scope.
- Task attrs: `notify`, single-tile `wait`, strided `wait`, and `trigger`.

CUDA local EventTensor lowering lives under `include/megacu/platform/cuda/`.
CUDA+NVSHMEM team EventTensor lowering lives under
`include/megacu/backends/nvshmem/`. The NVSHMEM implementation signals each
producer tile to all PEs and waits for all PEs for the requested tile. Whole
event waits remain available through `event_tensor::wait(event)`.

## Comparison With Event Tensor

Original Event Tensor presents a higher-level compiler/runtime model that can
generate many fine-grained event operations from a compact tiled program.
Megacu exposes the lower-level orchestration surface directly. The same
producer-event-consumer patterns can be represented in Megacu by submitting
tasks at the desired granularity and connecting them with sync-only
EventTensor tasks. The difference is authoring and generation burden, not a
loss of expressive power.

This is intentional for the current project stage. The implementation stays
thin, the ownership boundaries are inspectable, and target authors or later
generators remain responsible for choosing task granularity.
