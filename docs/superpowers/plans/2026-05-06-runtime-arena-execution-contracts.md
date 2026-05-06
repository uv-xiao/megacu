# Runtime Arena Execution Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first architecture checkpoint real: host-only and CUDA smoke contracts proving that the runtime loop consumes arena task records through scheduler, dispatcher, EventTensor, and operator-table components.

**Architecture:** This slice does not implement GEMM math. It locks the common execution path that every example must use before GEMM-AllReduce is refit: one shared recipe creates EventTensor records, operator tasks, and sync-only tasks; `host-orch` and `seeded-orch` produce equivalent topology; the runtime loop consumes the sealed/published arena rather than iterating hard-coded numeric task ids. EventTensor remains a linked component and is invoked from task attrs, not operator code.

**Tech Stack:** C++20 headers, CUDA `.cuh/.cu` smoke contracts, CMake/CTest, `assert` only if the current contract-test style is preserved for this slice.

---

## File Structure

- `include/megacu/runtime/task_arena.h`: extend arena records with completion state only if the existing fields cannot represent runtime execution.
- `include/megacu/runtime/device_types.cuh`: add device task/work status types if needed by the arena-consuming loop.
- `include/megacu/runtime/recipe_contracts.h`: create a host/device-friendly tiny recipe helper shared by host-only and CUDA smoke tests.
- `include/megacu/runtime/loop/block_tile.cuh`: change the loop from numeric scheduler iteration to arena-record execution.
- `include/megacu/scheduler/explicit_asap_device.cuh`: make device scheduler read published arena tasks and explicit deps.
- `include/megacu/dispatcher/tile_grid_device.cuh`: make device dispatcher derive work cursor availability from task dispatch attrs and task kind.
- `include/megacu/backends/nvshmem/cuda_event_tensor.cuh`: add a small attr-driven EventTensor adapter for smoke tests without hard-coded notify/wait task ids.
- `tests/build/runtime_arena_execution_contracts.cc`: new host-only topology/equivalence contract.
- `tests/build/runtime_arena_execution_contract.cu`: new CUDA runtime-loop smoke contract.
- `examples/cuda_nvshmem/CMakeLists.txt` or the current root CMake target registration location: add both new test targets.
- `docs/in_progress/general_runtime_linked_components_code_review.tmp.md`: update the component dashboard after the checkpoint lands.

## Contract Recipe

Both tests should use this topology:

```text
event ready = event_tensor(shape(1, 1), wait_count(1))
task producer = submit(op_slot{3}, event_tensor::notify(ready), dispatcher::tile_grid(1, 1))
task wait_ready = sync(depends_on(producer), event_tensor::wait(ready))
task consumer = submit(op_slot{4}, depends_on(wait_ready), dispatcher::tile_grid(1, 1))
seal epoch 0
```

Expected topology:

```text
events: 1
tasks: 3
deps: 2
regions: 1 sealed region
task 0: operator op_slot 3, no deps, notify ready
task 1: sync-only, depends on task 0, waits ready
task 2: operator op_slot 4, depends on task 1
```

Expected runtime execution order for one tile:

```text
scheduler ready task 0
dispatcher cursor active for task 0
operator slot 3 invoked
EventTensor sees notify(ready)
scheduler marks task 0 complete
scheduler ready task 1
EventTensor sees wait(ready) complete
sync task completes without invoking an operator
scheduler ready task 2
dispatcher cursor active for task 2
operator slot 4 invoked
scheduler marks task 2 complete
```

## Task 1: Host-Only Topology Contract

**Files:**
- Create: `include/megacu/runtime/recipe_contracts.h`
- Create: `tests/build/runtime_arena_execution_contracts.cc`
- Modify: current CMake test registration location for build contracts.

- [ ] **Step 1: Write the host-only failing contract**

Create `tests/build/runtime_arena_execution_contracts.cc` with this skeleton:

```cpp
#include <cassert>

#include <megacu/runtime/execution/host_orch.h>
#include <megacu/runtime/recipe_contracts.h>

int main() {
  megacu::runtime::execution::host_orch::frame<4, 2, 4> frame;
  auto refs = megacu::runtime::contracts::submit_three_task_event_recipe(frame);

  assert(refs.ready == megacu::runtime::event_tensor_ref{0});
  assert(refs.producer == megacu::runtime::task_ref{0});
  assert(refs.wait_ready == megacu::runtime::task_ref{1});
  assert(refs.consumer == megacu::runtime::task_ref{2});
  assert(frame.seal().code == megacu::status_code::ok);

  auto tasks = frame.tasks();
  auto events = frame.event_tensors();
  auto deps = frame.deps();
  auto regions = frame.regions();

  assert(events.size() == 1);
  assert(tasks.size() == 3);
  assert(deps.size() == 2);
  assert(regions.size() == 1);
  assert(regions[0].sealed == 1);

  assert(tasks[0].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[0].op_slot == megacu::runtime::operator_slot{3});
  assert(tasks[0].dep_count == 0);

  assert(tasks[1].kind == megacu::runtime::task_kind::sync_only);
  assert(tasks[1].op_slot == megacu::runtime::invalid_operator_slot());
  assert(tasks[1].dep_count == 1);
  assert(deps[tasks[1].first_dep] == refs.producer);

  assert(tasks[2].kind == megacu::runtime::task_kind::operator_body);
  assert(tasks[2].op_slot == megacu::runtime::operator_slot{4});
  assert(tasks[2].dep_count == 1);
  assert(deps[tasks[2].first_dep] == refs.wait_ready);
  return 0;
}
```

- [ ] **Step 2: Register and run the failing contract**

Add a build target named `megacu_runtime_arena_execution_contracts`.

Run:

```bash
cmake -S . -B build
cmake --build build --target megacu_runtime_arena_execution_contracts
```

Expected: FAIL because `megacu/runtime/recipe_contracts.h` is missing.

- [ ] **Step 3: Implement the shared recipe helper**

Create `include/megacu/runtime/recipe_contracts.h`:

```cpp
#pragma once

#include <megacu/runtime.h>
#include <megacu/runtime/task_arena.h>

namespace megacu::runtime::contracts {

struct three_task_event_refs {
  event_tensor_ref ready = invalid_event_tensor_ref();
  task_ref producer = invalid_task_ref();
  task_ref wait_ready = invalid_task_ref();
  task_ref consumer = invalid_task_ref();
};

template <class Orch>
MEGACU_RUNTIME_HOST_DEVICE three_task_event_refs
submit_three_task_event_recipe(Orch &orch) {
  three_task_event_refs refs;
  refs.ready = orch.event_tensor(attrs(event_tensor::shape(1, 1),
                                       event_tensor::wait_count(1)));
  refs.producer = orch.submit(
      operator_slot{3},
      attrs(dispatcher::tile_grid(1, 1), event_tensor::notify(refs.ready)));
  refs.wait_ready = orch.sync(attrs(scheduler::depends_on(refs.producer),
                                    event_tensor::wait(refs.ready)));
  refs.consumer = orch.submit(
      operator_slot{4},
      attrs(scheduler::depends_on(refs.wait_ready),
            dispatcher::tile_grid(1, 1)));
  return refs;
}

} // namespace megacu::runtime::contracts
```

If `MEGACU_RUNTIME_HOST_DEVICE` is not visible outside `runtime.h`, replace it with a local macro in this header:

```cpp
#if defined(__CUDACC__)
#define MEGACU_CONTRACTS_HOST_DEVICE __host__ __device__
#else
#define MEGACU_CONTRACTS_HOST_DEVICE
#endif
```

and use `MEGACU_CONTRACTS_HOST_DEVICE` on the function.

- [ ] **Step 4: Run the host-only contract**

Run:

```bash
cmake --build build --target megacu_runtime_arena_execution_contracts
ctest --test-dir build -R runtime_arena_execution_contracts --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/megacu/runtime/recipe_contracts.h tests/build/runtime_arena_execution_contracts.cc CMakeLists.txt examples/cuda_nvshmem/CMakeLists.txt cmake/MegacuTargets.cmake
git commit -m "test: add runtime arena topology contract"
```

Only add CMake files that were actually modified.

## Task 2: Arena-Consuming Device Scheduler And Dispatcher

**Files:**
- Modify: `include/megacu/runtime/task_arena.h`
- Modify: `include/megacu/runtime/device_types.cuh`
- Modify: `include/megacu/scheduler/explicit_asap_device.cuh`
- Modify: `include/megacu/dispatcher/tile_grid_device.cuh`
- Test: `tests/build/runtime_arena_execution_contract.cu`

- [ ] **Step 1: Write the CUDA failing contract for scheduler/dispatcher shape**

Create `tests/build/runtime_arena_execution_contract.cu` with a kernel that uses a sealed host-style arena copied into managed memory. The test should create three task records matching the recipe, one event record, two dependency records, and one sealed region. The kernel should call `scheduler.first(ctx, arena, work)` and verify task 0 is the first ready task before any completion state is marked.

Use a minimal observation struct:

```cpp
struct arena_runtime_observation {
  int first_task = -1;
  int first_work_active = 0;
  int first_work_tile = -1;
};
```

Expected initial kernel observation:

```cpp
assert(obs->first_task == 0);
assert(obs->first_work_active == 1);
assert(obs->first_work_tile == 0);
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
cmake --build build --target megacu_runtime_arena_execution_contract
```

Expected: FAIL because current `explicit_asap` does not accept or inspect arena records.

- [ ] **Step 3: Add minimal device completion state**

Extend `task_arena_view` only if necessary with:

```cpp
std::uint32_t *task_completed = nullptr;
```

If avoiding a struct change is cleaner, let the scheduler own a `std::uint32_t *task_completed` pointer instead. The chosen implementation must keep completion state runtime-owned, not example-owned.

- [ ] **Step 4: Make device scheduler read arena deps**

Change `megacu::scheduler::device::explicit_asap` so it exposes:

```cpp
template <class Context, class Arena, class Work>
__device__ megacu::runtime::device::task_ref first(Context, Arena arena,
                                                   Work work) const;

template <class Context, class Arena, class Work>
__device__ megacu::runtime::device::task_ref
next(Context, Arena arena, Work work,
     megacu::runtime::device::task_ref current) const;

template <class Context, class Arena, class Work>
__device__ void complete(Context, Arena arena,
                         megacu::runtime::device::task_ref task,
                         Work work) const;
```

Read only sealed region 0 for this checkpoint. A task is ready when:

```text
task id is within region task range
task_completed[task] == 0
all dependency task ids in arena.deps are completed
```

Keep the old overloads only if existing tests need compatibility.

- [ ] **Step 5: Make device dispatcher inspect task attrs**

Change `megacu::dispatcher::device::tile_grid` so it can produce a work item for a specific task:

```cpp
template <class Context, class Arena>
__device__ megacu::runtime::device::work_item first(Context ctx, Arena arena,
                                                    megacu::runtime::device::task_ref task) const;
```

For operator tasks, inspect `arena.tasks[task.value].attributes` for `attr_kind::dispatch_tile_grid`. Use `first * second` as the tile count. For sync-only tasks, return one active work item on block 0/thread-independent cursor so EventTensor can complete the sync task once.

- [ ] **Step 6: Run the scheduler/dispatcher CUDA contract**

Run:

```bash
cmake --build build --target megacu_runtime_arena_execution_contract
ctest --test-dir build -R runtime_arena_execution_contract --output-on-failure
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add include/megacu/runtime/task_arena.h include/megacu/runtime/device_types.cuh include/megacu/scheduler/explicit_asap_device.cuh include/megacu/dispatcher/tile_grid_device.cuh tests/build/runtime_arena_execution_contract.cu CMakeLists.txt examples/cuda_nvshmem/CMakeLists.txt
git commit -m "feat: read arena records in device scheduler and dispatcher"
```

Only add CMake files that were actually modified.

## Task 3: Runtime Loop Executes Arena Tasks

**Files:**
- Modify: `include/megacu/runtime/loop/block_tile.cuh`
- Modify: `tests/build/runtime_arena_execution_contract.cu`

- [ ] **Step 1: Extend CUDA contract to execute the whole topology**

In `runtime_arena_execution_contract.cu`, add:

```cpp
struct op_observation {
  int producer_count = 0;
  int consumer_count = 0;
  int sync_completed = 0;
  int task0_completed = 0;
  int task1_completed = 0;
  int task2_completed = 0;
};
```

Add an operator table:

```cpp
struct contract_operators {
  op_observation *obs = nullptr;

  template <class Context, class Task, class Work>
  __device__ void invoke(Context ctx, Task task, Work) const {
    if (ctx.thread_id() != 0) return;
    if (task.value == 0) atomicAdd(&obs->producer_count, 1);
    if (task.value == 2) atomicAdd(&obs->consumer_count, 1);
  }
};
```

Expected after runtime execution:

```cpp
assert(obs->producer_count == 1);
assert(obs->consumer_count == 1);
assert(obs->sync_completed == 1);
assert(completed[0] == 1);
assert(completed[1] == 1);
assert(completed[2] == 1);
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
cmake --build build --target megacu_runtime_arena_execution_contract
ctest --test-dir build -R runtime_arena_execution_contract --output-on-failure
```

Expected: FAIL because `runtime::loop::block_tile` still drives old numeric scheduler/dispatcher hooks and invokes operators for sync-only tasks.

- [ ] **Step 3: Update `runtime::loop::block_tile`**

Change the loop to:

```text
wait for region 0 sealed
for each dispatcher work cursor:
  ask scheduler for first ready task from arena
  while task valid:
    if task is sync-only:
      if EventTensor does not report completion, try next task
      mark task complete through scheduler
    else:
      ask dispatcher whether this worker has a cursor for this task
      invoke operator table
      ask EventTensor to process notify attrs
      mark task complete through scheduler
    task = scheduler.next(...)
```

The C++ shape should keep the component calls explicit:

```cpp
if (record.kind == megacu::runtime::task_kind::sync_only) {
  if (event_tensor.complete(ctx, arena, task, work)) {
    scheduler.complete(ctx, arena, task, work);
  }
  continue;
}

auto cursor = dispatcher.first(ctx, arena, task);
if (!cursor.active) continue;
operators.invoke(ctx, task, cursor);
event_tensor.after(ctx, arena, task, cursor);
scheduler.complete(ctx, arena, task, cursor);
```

Do not add phase ordering. Do not infer dependencies from tensor arguments.

- [ ] **Step 4: Run the CUDA execution contract**

Run:

```bash
cmake --build build --target megacu_runtime_arena_execution_contract
ctest --test-dir build -R runtime_arena_execution_contract --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/megacu/runtime/loop/block_tile.cuh tests/build/runtime_arena_execution_contract.cu
git commit -m "feat: execute arena tasks in block tile runtime loop"
```

## Task 4: Attr-Driven EventTensor Smoke Component

**Files:**
- Modify: `include/megacu/backends/nvshmem/cuda_event_tensor.cuh`
- Modify: `tests/build/runtime_arena_execution_contract.cu`

- [ ] **Step 1: Extend CUDA contract to fail on hard-coded task ids**

In the CUDA contract, instantiate EventTensor without notify/wait task-id fields. The test must pass only if the EventTensor component scans task attrs:

```cpp
megacu::backend::nvshmem::cuda::attr_event_tensor_i32 event_tensor{
    .events = event_storage,
    .tiles = 1,
    .my_pe = 0,
    .n_pes = 1};
```

The test should assert:

```cpp
assert(event_storage[0] == 1);
assert(obs->sync_completed == 1);
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
cmake --build build --target megacu_runtime_arena_execution_contract
ctest --test-dir build -R runtime_arena_execution_contract --output-on-failure
```

Expected: FAIL because only `task_event_tensor_i32` supports hard-coded `notify_task` and `wait_task`.

- [ ] **Step 3: Add attr-driven EventTensor adapter**

Add a new type instead of deleting the old one in this slice:

```cpp
struct attr_event_tensor_i32 {
  event_tensor_i32 event_tensor;

  template <class Context, class Arena, class Task, class Work>
  __device__ bool complete(Context, Arena arena, Task task, Work work) const;

  template <class Context, class Arena, class Task, class Work>
  __device__ void after(Context, Arena arena, Task task, Work work) const;
};
```

`complete(...)` returns true for sync-only tasks with no wait attrs. For each `event_wait` attr, it waits on the event represented by the attr and returns true after the wait completes.

`after(...)` scans `event_notify` attrs and notifies those events after an operator task finishes.

For this smoke contract, map event ref 0 and tile 0 to `event_storage[0]`. Keep distributed NVSHMEM behavior delegated to the existing `event_tensor_i32::notify/wait` functions.

- [ ] **Step 4: Run the CUDA execution contract**

Run:

```bash
cmake --build build --target megacu_runtime_arena_execution_contract
ctest --test-dir build -R runtime_arena_execution_contract --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/megacu/backends/nvshmem/cuda_event_tensor.cuh tests/build/runtime_arena_execution_contract.cu
git commit -m "feat: lower EventTensor operations from task attrs"
```

## Task 5: Seeded-Orch Equivalence CUDA Smoke

**Files:**
- Modify: `tests/build/runtime_arena_execution_contract.cu`
- Modify: `include/megacu/runtime/recipe_contracts.h` if the recipe helper needs CUDA annotations.

- [ ] **Step 1: Extend CUDA test to build topology through `seeded-orch`**

Add a `seeded_contract_recipe` that calls `submit_three_task_event_recipe(orch)` and then `orch.seal()`.

Launch a `device_persistent<seeded_orch::model<seeded_contract_recipe>, runtime::loop::block_tile, scheduler::device::explicit_asap, dispatcher::device::tile_grid, attr_event_tensor_i32, contract_operators>` runtime.

- [ ] **Step 2: Assert seeded-orch runtime behavior**

After the kernel:

```cpp
assert(device_status->code == megacu::status_code::ok);
assert(construction_status[0] ==
       megacu::runtime::execution::seeded_orch::construction_succeeded);
assert(tasks[0].op_slot == megacu::runtime::operator_slot{3});
assert(tasks[1].kind == megacu::runtime::task_kind::sync_only);
assert(tasks[2].op_slot == megacu::runtime::operator_slot{4});
assert(obs->producer_count == 1);
assert(obs->sync_completed == 1);
assert(obs->consumer_count == 1);
```

- [ ] **Step 3: Run the CUDA smoke contract**

Run:

```bash
cmake --build build --target megacu_runtime_arena_execution_contract
ctest --test-dir build -R runtime_arena_execution_contract --output-on-failure
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/build/runtime_arena_execution_contract.cu include/megacu/runtime/recipe_contracts.h
git commit -m "test: prove seeded-orch arena execution smoke"
```

## Task 6: Review Doc Status Update

**Files:**
- Modify: `docs/in_progress/general_runtime_linked_components_code_review.tmp.md`

- [ ] **Step 1: Update dashboard statuses**

Change these rows if the previous tasks pass:

```text
Task/Event records: Partial -> Mostly complete
Runtime loops: Partial -> Mostly complete
Scheduler, device side: Partial -> Mostly complete
Dispatcher, device side: Partial -> Mostly complete
EventTensor API: Mostly complete -> Mostly complete with CUDA smoke lowering
CUDA+NVSHMEM EventTensor lowering: Partial -> Partial, attr-driven smoke added
```

- [ ] **Step 2: Add verification evidence**

Add the exact commands and outcomes:

```text
cmake --build build --target megacu_runtime_arena_execution_contracts
ctest --test-dir build -R runtime_arena_execution_contracts --output-on-failure
cmake --build build --target megacu_runtime_arena_execution_contract
ctest --test-dir build -R runtime_arena_execution_contract --output-on-failure
```

- [ ] **Step 3: Run stale-term and section checks**

Run:

```bash
rg -n "overlap|co-resident|co_resident|co-residency|co-resilience|co resilience|retired concurrency topic|retired residency" docs/in_progress/general_runtime_linked_components_code_review.tmp.md -S
rg -n "^##|^###" docs/in_progress/general_runtime_linked_components_code_review.tmp.md
```

Expected: first command returns no matches; second command shows `Architecture Overview`, `Next Step Decision Record`, and findings still present.

- [ ] **Step 4: Commit**

```bash
git add docs/in_progress/general_runtime_linked_components_code_review.tmp.md
git commit -m "docs: record runtime arena execution checkpoint status"
```

## Execution Notes For Subagents

- Do not implement GEMM-AllReduce in this plan. This plan ends when the contract path is real.
- Do not preserve a compatibility path by changing the new contract to match old numeric scheduler behavior.
- Do not add fallback validation logic to production runtime paths. Tests may assert shape and behavior.
- Do not make EventTensor a second scheduler. `depends_on` controls task readiness; EventTensor controls sync-task completion.
- Do not use CUDA streams for internal task synchronization. Streams only belong to host enqueue/final synchronization.
- Do not use problem-specific phases or phase priority.
- Do not let operator bodies call scheduler, dispatcher, EventTensor wait/notify, CUDA stream sync, or NVSHMEM sync directly.
- Do not dispatch multiple implementation subagents in parallel against the same headers. These tasks are sequential because they touch the same runtime files.

## Self-Review

Spec coverage:

- Covers the agreed first checkpoint: host-only topology contract and CUDA smoke.
- Covers same recipe for `host-orch` and `seeded-orch`.
- Covers arena consumption by runtime loop.
- Covers scheduler readiness from explicit dependencies.
- Covers dispatcher cursor from task attrs.
- Covers EventTensor attr-driven notify/wait smoke without manual task ids.
- Does not cover GEMM-AllReduce refit; that is the next plan after this checkpoint.

Placeholder scan:

- No placeholder markers or vague "add tests" instructions are intentionally
  present.

Type consistency:

- Uses existing names: `task_arena_view`, `operator_slot`, `task_ref`, `event_tensor_ref`, `host_orch::frame`, `seeded_orch::model`, `runtime::loop::block_tile`, `scheduler::device::explicit_asap`, `dispatcher::device::tile_grid`.
