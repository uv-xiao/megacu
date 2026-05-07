# Feature Task: General Runtime-Linked Components

- Branch: `implementation/general-runtime-linked-components`
- PR: TBD
- Owner: Megacu agents
- Status: active

## Goal

Make the runtime-linked implementation more complete, general, and aligned with
the accepted architecture after PR #4. This PR must move beyond the tiny
GEMM+AllReduce proof while preserving the thin design: user code submits raw
operator calls with explicit attributes, and linked dispatcher, scheduler,
backend, platform, launch-adapter, and runtime strategies do the rest.
The PR example scope is GEMM-RS, AG-GEMM, and a tiny decode pipeline. The
Megacu examples must be composed from tile/range operators submitted as tasks,
not written as one problem-specific manual mega-kernel.

## Input

- Implemented PR #4 runtime-linked slice.
- Stable design under `docs/design/runtime_linked_device_native_layer/`.
- Future concrete implementation notes under `docs/todo/concrete_impl/`.
- Human correction that MPI and Torch adapters are required PR scope and must
  execute through Docker.

## Output

- Required Docker-backed CUDA+NVSHMEM development/test environment with MPI and
  PyTorch available.
- MPI launch adapter that creates the same CUDA+NVSHMEM driver shape from an
  MPI-launched world.
- Torch launch adapter that creates the same CUDA+NVSHMEM driver shape from a
  `torchrun`/`torch.distributed` world.
- Multiple linked dispatcher, scheduler, runtime execution-model, and
  runtime-loop choices selected by target/configuration, not programmed by
  normal users.
- Additional examples/design input from Triton-distributed tutorial 07 and
  tutorial 08, recorded in `docs/notes/general_runtime_examples_sources.md`.
- A GEMM-RS example with separate golden, baseline, and Megacu paths.
- An AG-GEMM example with separate golden, baseline, and Megacu paths.
- A Hazy-style end-to-end tiny decode pipeline example with separate golden,
  baseline, and Megacu paths.
- Build/runtime contract tests proving the shared
  dispatcher/scheduler/runtime APIs are reusable across the three examples.

## Scope Checklist

- [x] Define input, output, and verification criteria
- [x] Write or update design docs if architecture changes
- [ ] Implement in coherent commits
- [ ] Verify locally and through Docker
- [ ] Sync `docs/design/`, `docs/todo/`, and `docs/in_progress/`

## Verification

Required before PR publication:

- `git diff --check`
- local CMake configure/build/test suite
- Docker build with CUDA, NVSHMEM, MPI, and PyTorch dependencies
- Docker direct/local GEMM-RS correctness
- Docker MPI-launched GEMM-RS correctness
- Docker `torchrun`/`torch.distributed` GEMM-RS correctness
- Docker direct/local and distributed AG-GEMM correctness
- Docker tiny decode pipeline correctness comparing Megacu, baseline, and
  golden
- documented dispatcher/scheduler/runtime candidate review before implementation
- build tests proving more than one dispatcher, scheduler, runtime execution
  model, and runtime loop can be selected without changing normal user
  orchestrate code
- grep or compile guards proving users do not call `scheduler.run`, launch
  adapters do not choose placement/order, and the rejected materializer path
  does not return

## Tests

- Keep PR #4 runtime and build tests.
- Add adapter smoke tests for MPI and Torch.
- Add strategy-selection contract tests.
- Add at least one non-GEMM reuse test.
- Add tiny decode pipeline runtime tests for golden, baseline, and Megacu.

## Implementation Plan

### Slice 1: EventTensor API Decoupling

**Goal:** Make the checked-in API match the accepted EventTensor design before
larger runtime-strategy work.

**Files:**

- Modify `include/megacu/runtime.h` to expose
  `megacu::runtime::event_tensor::{shape,wait_count,notify,wait,trigger}` and
  remove normal-use references to `runtime::events::{publish,acquire,join}`.
- Modify `include/megacu/backends/nvshmem/cuda_event_tensor.cuh` to use
  `notify` and `wait` naming for lowered operations.
- Modify `src/dispatcher/explicit_attrs.cc` and runtime contract counters from
  publish/acquire/join to notify/wait/trigger.
- Modify `tests/build/runtime_linked_surface_contract.cc` and
  `tests/build/runtime_components_contracts.cc` first so the desired API fails
  before implementation.
- Modify the CUDA+NVSHMEM Megacu example to construct the renamed EventTensor
  component.
- Add or update grep tests in
  `examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt` to reject stale
  public event API names in Megacu code.

**Verification:**

- `cmake --build build --target megacu_runtime_components_contracts megacu_runtime_linked_surface_contract`
- `ctest --test-dir build -R 'runtime_components_contracts|runtime_linked_surface_contract|megacu_event_tensor' --output-on-failure`
- `git diff --check`
- Documentation policy grep from this task.

**Status:** implemented in this branch. The public API now uses
`event_tensor::notify`, sync-task `event_tensor::wait`, and decoupled
`cuda_nvshmem::event_tensor` storage attrs.

### Slice 2: Runtime Loop Naming

**Goal:** Replace the stale device-entry naming with the accepted runtime
loop vocabulary.

**Files:**

- Create `include/megacu/runtime/block_tile_runtime.cuh`.
- Remove `include/megacu/runtime/device_entry.cuh`.
- Modify the CUDA+NVSHMEM Megacu example to instantiate
  `megacu::runtime::device::block_tile_runtime`.
- Update the CTest guard so the example and runtime headers reject
  `device_entry.cuh` and `runtime::device::entry`.

**Verification:**

- `cmake -S . -B build`
- `cmake --build build --target cuda_nvshmem_gemm_allreduce_phased`
- `ctest --test-dir build -R 'megakernel_running_logic_is_common|runtime_components_contracts|runtime_linked_surface_contract|megacu_event_tensor' --output-on-failure`

**Status:** implemented in this branch. The loop behavior is unchanged, but the
component now matches the runtime-loop architecture.

### Slice 3: Runtime Execution Models

**Goal:** Implement runtime-owned execution models for `host-orch` and
`seeded-orch`, with both validated through examples.

**Files:**

- Add runtime execution-model headers under `include/megacu/runtime/execution/`.
- Add compact arena record headers under `include/megacu/runtime/`.
- Add or rename the loop path under `include/megacu/runtime/loop/`.
- Add `runtime::device_persistent<ExecutionModel, Loop, ...>` composition.
- Modify the CUDA+NVSHMEM Megacu example so host-orch and seeded-orch use the
  same target recipe and operator table.
- Add build/runtime tests proving both execution models share scheduler,
  dispatcher, EventTensor, platform, and backend contracts.

**Verification:**

- `cmake --build build --target megacu_runtime_components_contracts`
- new runtime execution-model build test
- CUDA+NVSHMEM host-orch example for one host/one GPU and one host/two GPUs
- CUDA+NVSHMEM seeded-orch example for one host/one GPU and one host/two GPUs
- grep guard rejecting top-level `build_run::` in code and examples

**Status:** design recorded in
`docs/in_progress/design/runtime_execution_model_implementation_design.md`;
implementation in progress. Compact arena records, `host_orch::frame`,
`seeded_orch`, runtime composition, GEMM-AllReduce runtime integration,
GEMM-RS host-orch/seeded-orch local CUDA integration, AG-GEMM
host-orch/seeded-orch local CUDA integration, and tiny decode host-orch/
seeded-orch local CUDA integration now exist. Distributed GEMM-RS, distributed
AG-GEMM, and distributed tiny decode integration remain pending.

### Slice 4: Three Tile-Operator Examples

**Goal:** Implement the PR example scope with Megacu-composed operator tasks,
not problem-specific manual mega-kernels.

**Files:**

- Add `examples/cuda_nvshmem/gemm_reduce_scatter/`.
- Add `examples/cuda_nvshmem/allgather_gemm/`.
- Add `examples/cuda_nvshmem/tiny_decode_pipeline/`.
- Keep any handwritten mega-kernel only under `baseline/`.
- Keep Megacu variants under `megacu/` and route them through linked runtime,
  scheduler, dispatcher, EventTensor, platform, and backend components.

**Verification:**

- Docker direct/local GEMM-RS correctness.
- Docker MPI-launched GEMM-RS correctness.
- Docker `torchrun`/`torch.distributed` GEMM-RS correctness.
- Docker direct/local and distributed AG-GEMM correctness.
- Docker tiny decode golden/baseline/Megacu correctness.
- Grep/manual review proving Megacu variants submit tile/range operator tasks
  rather than coding one problem-specific manual mega-kernel.

## Docs

- Active design:
  `docs/in_progress/design/general_runtime_linked_components.md`
- Overall architecture draft:
  `docs/in_progress/design/overall_runtime_architecture.md`
- Runtime execution-model implementation design:
  `docs/in_progress/design/runtime_execution_model_implementation_design.md`
- Stable docs recreation plan:
  `docs/in_progress/design/stable_docs_recreation_plan.md`
- Source notes:
  `docs/notes/general_runtime_examples_sources.md`
- Stable architecture docs should not be incrementally appended during active
  implementation. Before merge, recreate the stable `docs/design/` package from
  the accepted active design and implemented behavior. The current stable PR #4
  folder should be removed and replaced with flat design docs for Megacu users
  and contributors.

## Closeout

- Update PR body with exact Docker commands and evidence.
- Recreate stable design docs according to
  `docs/in_progress/design/stable_docs_recreation_plan.md`.
- Remove or mark completed this task file and active design draft.
