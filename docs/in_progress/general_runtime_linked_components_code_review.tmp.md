# Temporary Review: General Runtime-Linked Components

Date: 2026-05-06 Asia/Shanghai

Branch reviewed: `implementation/general-runtime-linked-components`

Reviewed range: `main..HEAD`

Purpose: compare the current implementation against the active design under
`docs/in_progress/design/`, especially runtime ownership, EventTensor semantics,
required adapters, required examples, and verification status.

This is a temporary review artifact for human review. It is not a stable design
document.

## Architecture Overview And Completion Dashboard

Status key:

- Complete: implemented and locally verified for this PR slice.
- Mostly complete: implemented structurally, with limited integration gaps.
- Partial: API or contract exists, but it is not fully wired through examples,
  runtime execution, or distributed validation.
- Missing: required by the active design, but not implemented beyond a stub.
- Process/deferred: intentionally left for PR closeout or unavailable
  environment verification.

Architecture flow under the active design:

1. Host code calls `orchestrate(driver, raw kernel-like args...)`. Megacu should
   avoid wrapping user buffers; arguments should remain close to CUDA kernel
   arguments.
2. The orchestrator records operator tasks and sync-only tasks. User-visible
   dependencies are explicit scheduling attributes, not inferred tensor
   read/write analysis.
3. Each submitted task carries compact attrs for scheduling, dispatch, operator
   identity, and EventTensor behavior. `submit`/`sync` should inject the
   built-in operator/sync attrs instead of requiring operator code to hand-roll
   event protocol.
4. The runtime owns the build/run model and the internal loop. Different
   runtimes can implement different loops, such as direct issue, spin-wait for
   spatial resources, or buffering ready-but-not-issued work.
5. The scheduler owns task readiness from explicit dependencies. It should not
   know GEMM-RS, AG-GEMM, tiny decode, or problem-specific phase order.
6. The dispatcher owns spatial placement and work cursors from task type and
   dispatch attrs. It should decide which grid/block/tile executes the selected
   task.
7. EventTensor owns sync task completion conditions and platform/backend
   lowering. It co-exists with `depends_on`: `depends_on` decides when a task is
   schedulable; EventTensor decides when a sync task has actually completed.
8. Platform/backend/driver code binds this model to CUDA+NVSHMEM. The driver
   should carry launch, rank, team, and communication facts needed by the common
   runtime while keeping MPI/Torch as required launch adapter paths.
9. Examples should demonstrate the model by composing small operator tasks into
   one Megacu runtime path, then comparing against golden and handwritten
   baselines in single-device and distributed runs.

Component status:

| Area | Design role | Current implementation | Status | Evidence |
| --- | --- | --- | --- | --- |
| Public authoring surface | Host-callable orchestration with raw kernel-like arguments and explicit attrs | `runtime::phase`, `submit`, `sync`, and attrs exist, but examples do not all use the intended end-to-end path | Partial | Runtime headers and tests compile in the previous implementation pass |
| Attr system | Shared carrier for scheduling, dispatch, operator, and EventTensor metadata | Central enum/union attr path exists; component-owned attr extensibility is still limited | Partial | Attr contract tests and runtime headers |
| Task/Event records | Compact records for host-orch and seeded-orch execution | Arena and record types exist, but runtime loops do not yet consume them as the primary execution source | Partial | `task_arena` contracts |
| `host-orch` model | Host builds task records, runtime launches a snapshot | Frame contracts exist | Partial | Host-orch contract tests, not required examples |
| `seeded-orch` model | Host seeds device arena; persistent runtime can build/consume work dynamically | Device publication contracts exist | Partial | CUDA contract proof, not end-to-end examples |
| Runtime composition | Link scheduler, dispatcher, EventTensor, and operators behind a common runtime API | `device_persistent<ExecutionModel, Loop, ...>` provides a composition point | Mostly complete | Compile contracts for loop candidates |
| Runtime loops | Overrideable internal loops owned by runtime | `block_tile`, `grid_stride`, and `single_work_item` candidates exist, but do not drive real arena task execution | Partial | Strategy headers and tests |
| Scheduler, host side | ASAP readiness from explicit dependencies only | `explicit_asap` exists as a host-side contract | Partial | Contract tests |
| Scheduler, device side | Common task readiness for runtime loops | Device path is still effectively a numeric iterator, not arena dependency readiness | Partial | Runtime loop implementation |
| Dispatcher, host side | Decode dispatch attrs into work cursor/placement decisions | Attr counter/contract path exists | Partial | Dispatcher tests |
| Dispatcher, device side | Pick spatial work for selected ready task | `tile_grid` exists, but remains a narrow candidate | Partial | Device dispatcher headers |
| EventTensor API | Event shape plus notify/wait/trigger attrs used by sync tasks | API and attr contracts exist | Mostly complete | EventTensor tests and docs |
| CUDA+NVSHMEM EventTensor lowering | Backend/platform implementation for sync completion | Existing proof path still couples event lowering to manual task ids | Partial | GEMM+AllReduce proof path finding below |
| Driver | Carry CUDA+NVSHMEM launch/rank/team facts into runtime | Short driver view exists; conceptually acceptable, but more distributed fields may be needed as examples become real | Partial | CUDA+NVSHMEM driver/adapters |
| MPI adapter | Required distributed launch path | Normalizes MPI facts into driver facts | Partial | Adapter contracts; no full example run evidence |
| Torch adapter | Required distributed launch path | Normalizes torchrun environment into driver facts | Partial | Adapter contracts; no full example run evidence |
| Docker environment | Required execution envelope for MPI/Torch/CUDA+NVSHMEM | Helper scripts exist, but do not prove all required examples | Partial | Docker/tooling findings below |
| GEMM-RS example | Required end-to-end distributed example | Directory/build skeleton exists | Missing | No golden/baseline/Megacu correctness binary |
| AG-GEMM example | Required end-to-end distributed example | Directory/build skeleton exists | Missing | No golden/baseline/Megacu correctness binary |
| Tiny decode example | Required end-to-end decode pipeline example | Golden/baseline/Megacu host-phase correctness path exists | Partial | Not yet runtime-owned CUDA path |
| Stable docs closeout | Recreate flattened `docs/design/` at merge time | Active in-progress docs record this requirement | Process/deferred | Stable docs were touched early; see finding |

Task completion status:

| Task/design area | Target requirement | Current proof | Status |
| --- | --- | --- | --- |
| Architecture documentation | Active design explains runtime, scheduler, dispatcher, EventTensor, adapters, examples, and closeout docs | `docs/in_progress/design/` contains the discussion and implementation spec | Mostly complete |
| Runtime/EventTensor API rename and decoupling | Use `runtime`, not `entry`; use `EventTensor`, not EventEngine; keep EventTensor as task attrs, not operator-manual protocol | Headers/docs/tests reflect the rename | Mostly complete |
| Explicit task dependency model | Dependencies are explicit attrs and scheduler-owned | Host scheduler contracts exist | Partial |
| Runtime-owned internal loop | Runtime owns loop strategies, not scheduler or examples | Loop strategy candidates compile | Partial |
| Host-orch execution model | Snapshot-style host built task plan | Contract exists | Partial |
| Seeded-orch execution model | Device-resident orchestration seeded from host | Device publication contract exists | Partial |
| Scheduler/dispatcher strategy set | Provide multiple strategies that can be picked through the common runtime API | Some runtime loops exist; scheduler/dispatcher candidates are thin | Partial |
| MPI/Torch required adapters | Required by PR and Docker path, not optional dependency paths | Adapter normalization exists | Partial |
| GEMM-RS | Tile-operator Megacu example with golden and baseline comparison | Skeleton only | Missing |
| AG-GEMM | Tile-operator Megacu example with golden and baseline comparison | Skeleton only | Missing |
| Tiny decode | Pipeline example with golden, baseline, and Megacu comparison | Host-phase path exists | Partial |
| Docker verification | Build/run required distributed examples through Docker | Scripts do not yet prove required examples | Missing |
| `docs/design/` closeout | Recreate flattened stable docs during PR merge, not now | In-progress docs mention this | Process/deferred |

How to read the rest:

- `Executive Status` gives the short merge-readiness answer.
- `Findings` lists concrete mismatches and risks, ordered by severity.
- `Design Correspondence Matrix` maps design requirements to current code.
- `Recommended Fix Order` gives the implementation order I would use next.

## Next Step Decision Record

Decisions confirmed during the follow-up review discussion:

- The next step is to make this PR mergeable for the stated active design, not
  to narrow it to a smaller architecture prototype.
- The PR is complete only when all four examples work through the frozen
  architecture:
  - GEMM-AllReduce;
  - GEMM-RS;
  - AG-GEMM;
  - tiny decode pipeline.
- GEMM-AllReduce must be refit to the same operator-task plus runtime
  execution/loop model as the newer examples. It is not allowed to remain a
  Megacu-named legacy proof path with a problem-specific manual mega-kernel.
- Existing legacy proof code that violates the common runtime model must be
  removed. Handwritten kernels are allowed only as clean baselines under
  `baseline/`; they must not use Megacu runtime, scheduler, dispatcher, task, or
  EventTensor concepts.
- The Megacu variant for each example must be composed from operator tasks and
  sync-only tasks. The runtime owns the internal loop; examples do not program
  problem-specific runtime loops.
- `host-orch` and `seeded-orch` are required for all four examples. Each
  example should use one shared recipe shape, with only the runtime execution
  model changing:
  `recipe(args, orch) -> EventTensors -> submit operator tasks -> sync-only
  tasks -> sealed/published arena -> runtime loop`.
- Every example must provide golden, baseline, Megacu `host-orch`, and Megacu
  `seeded-orch` variants.
- Every example must run for 1-host-1-device and 1-host-2-device. Tiny decode
  specifically needs both versions, not only a local path.
- Every example must exercise direct, MPI, and Torch launch paths. For
  distributed validation, MPI and Torch must run the 1-host-2-device path in
  Docker.
- The required strategy set for this PR is:
  - dispatchers: `tile_grid`, `rank_aware_tile_grid`, `single_work_item`;
  - schedulers: `explicit_asap`, `static_submission_order`,
    `static_level_order`;
  - runtime execution models: `host-orch`, `seeded-orch`;
  - runtime loops: `block_tile_runtime`, `single_work_item_runtime`,
    `grid_stride_runtime`;
  - EventTensor: one thin CUDA+NVSHMEM counter/signal-style implementation
    selected through common EventTensor attrs.
- The implementation should proceed depth-first through GEMM-AllReduce as the
  tracer bullet, then reuse the same runtime path for GEMM-RS, AG-GEMM, and
  tiny decode.
- Before refitting GEMM-AllReduce math kernels, implement the
  `runtime_arena_execution_contracts` checkpoint with both a host-only contract
  test and a CUDA smoke test.

The first coding checkpoint must prove these invariants:

- `host-orch` and `seeded-orch` can use the same recipe shape;
- `submit` injects operator-task attrs automatically;
- `sync` injects sync-task attrs automatically;
- `depends_on` is scheduler-owned;
- EventTensor `wait`/`notify` attrs are runtime/EventTensor-owned;
- scheduler readiness uses explicit dependencies only;
- dispatcher chooses work cursors from dispatch attrs and task kind;
- the runtime loop consumes arena task records and asks components in this
  order: scheduler readiness, EventTensor sync-task completion, dispatcher
  cursor availability, operator or sync handler, completion update;
- operator bodies receive raw args plus a thin operator context, not scheduler
  or EventTensor internals;
- task completion state is written through the arena;
- `host-orch` and `seeded-orch` produce equivalent task/event topology for the
  same recipe.

## Review Commands

- `git status --short --branch`
- `git show --stat --oneline main..HEAD`
- `git diff --name-only main..HEAD -- docs/design docs/in_progress/design`
- `rg` scans for design terms, forbidden stale terms, strategy names, adapter
  paths, runtime execution models, and example skeleton markers.
- Direct file reads with line numbers across `docs/in_progress/design/`,
  runtime headers, example code, adapter code, CMake, scripts, and tests.

Skipped during review:

- No build or CTest was rerun as part of this review pass.
- No Docker execution was run.
- No GitHub state was changed.

The previous implementation pass reported a local targeted build and CTest
result of 18/18 passing, but this review treats Docker and full distributed
correctness evidence as missing unless implemented and documented in code.

## Executive Status

Current implementation status: partial architecture slice, not merge-complete
for the full active design.

What is implemented:

- Runtime/EventTensor public API rename and basic attr contracts.
- Compact host/device task arena record types.
- `host-orch` frame and `seeded-orch` device publication prototypes.
- `runtime::device_persistent<ExecutionModel, Loop, ...>` composition point.
- Runtime loop candidates as compile/runtime contracts:
  `block_tile`, `grid_stride`, and `single_work_item`.
- Thin MPI/Torch launch fact adapters that normalize rank/world/local-rank into
  `cuda_nvshmem::driver_view`.
- Tiny decode golden/baseline/Megacu host-phase correctness path.
- GEMM-RS and AG-GEMM directories and buildable skeleton object targets.

What is not implemented relative to the active design:

- GEMM-RS and AG-GEMM are not end-to-end examples; they are skeletons.
- No GEMM-RS/AG-GEMM golden, baseline, or Megacu correctness binaries exist.
- No direct/MPI/Torch GEMM-RS correctness path exists.
- No direct/distributed AG-GEMM correctness path exists.
- `host-orch` and `seeded-orch` are not integrated into the required examples.
- The common runtime loop does not consume task arena records to decide
  scheduler/dispatcher/EventTensor behavior.
- EventTensor device lowering is still tied to hard-coded task ids in the
  GEMM+AllReduce proof path.
- Required dispatcher/scheduler strategy candidates are not all implemented.
- Docker helper scripts exercise adapter contracts and old GEMM+AllReduce tests,
  not the required new PR examples.

Merge-readiness assessment:

- Ready as a partial internal contract/prototype PR only if the PR scope is
  explicitly narrowed.
- Not ready for the stated active PR goal in
  `docs/in_progress/design/general_runtime_linked_components.md`.

## Findings

### High: GEMM-RS And AG-GEMM Are Skeletons, Not Required End-To-End Examples

Design requirement:

- `docs/in_progress/design/general_runtime_linked_components.md:6-8` says this
  PR must support three end-to-end examples: GEMM-RS, AG-GEMM, and tiny decode.
- `docs/in_progress/design/general_runtime_linked_components.md:66-71` says
  GEMM-RS and AG-GEMM must be written as operator tasks, sync-only tasks,
  explicit dependencies, dispatcher attrs, and EventTensor attrs.
- `docs/in_progress/design/general_runtime_linked_components.md:498-514`
  describes the required GEMM-RS and AG-GEMM task flows.
- `docs/in_progress/design/general_runtime_linked_components.md:849-853`
  requires Docker direct/MPI/Torch GEMM-RS correctness, direct/distributed
  AG-GEMM correctness, and tiny decode correctness.

Implementation evidence:

- `examples/cuda_nvshmem/gemm_reduce_scatter/megacu/gemm_reduce_scatter_megacu.cu:1-3`
  contains only a skeleton function returning `1`.
- `examples/cuda_nvshmem/allgather_gemm/megacu/allgather_gemm_megacu.cu:1-3`
  contains only a skeleton function returning `1`.
- `examples/cuda_nvshmem/gemm_reduce_scatter/README.md:63-70` explicitly says
  the current skeleton does not implement numeric correctness and does not add a
  numeric GEMM-RS distributed run.
- `examples/cuda_nvshmem/allgather_gemm/README.md:134-141` says the same for
  AG-GEMM.

Impact:

- The current branch does not prove reusable distributed task composition.
- The main examples that motivated this PR do not exercise scheduler,
  dispatcher, EventTensor, runtime execution model, or backend behavior.

Fix direction:

- Implement `common/`, `golden/`, `baseline/`, and `megacu/` for GEMM-RS and
  AG-GEMM.
- Add runnable correctness binaries for direct, MPI, and Torch launch paths.
- The Megacu variants should submit producer tasks, sync-only EventTensor wait
  tasks, and consumer tasks with explicit dependency attrs.

### High: `host-orch` And `seeded-orch` Exist Only As Contracts, Not Example Paths

Design requirement:

- `docs/in_progress/design/runtime_execution_model_implementation_design.md:8-13`
  requires `host-orch` and `seeded-orch` in this PR.
- `docs/in_progress/design/runtime_execution_model_implementation_design.md:287-288`
  requires `host-orch` and `seeded-orch` to produce equivalent task/event
  structure for the same recipe.
- `docs/in_progress/design/runtime_execution_model_implementation_design.md:290-328`
  requires host-orch and seeded-orch variants for GEMM-RS, AG-GEMM, and tiny
  decode, or at least explicit target names if a flatter layout is kept.
- `docs/in_progress/design/runtime_execution_model_implementation_design.md:418-423`
  requires Docker correctness comparison for golden, baseline, host-orch, and
  seeded-orch across GEMM-RS, AG-GEMM, and tiny decode.

Implementation evidence:

- `include/megacu/runtime/execution/host_orch.h` implements a host frame.
- `include/megacu/runtime/execution/seeded_orch.cuh` implements a seeded device
  construction model.
- `tests/build/runtime_execution_models_contracts.cc` exercises `host_orch`.
- `tests/build/runtime_device_persistent_contract.cu` exercises
  `seeded_orch`.
- `rg` shows `host_orch` and `seeded_orch` are used in tests and headers, but
  not in the new example targets.
- `examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_megacu.cu:279-304`
  uses host-side `runtime::phase`, not `runtime::execution::host_orch` or
  `runtime::execution::seeded_orch`.
- GEMM-RS and AG-GEMM Megacu sources are skeletons.

Impact:

- The core ownership claim, that examples are built through runtime execution
  models rather than manual task assembly or host-only phase execution, is not
  demonstrated.

Fix direction:

- Add example targets named around the accepted variants, for example
  `cuda_nvshmem_gemm_rs_host_orch`, `cuda_nvshmem_gemm_rs_seeded_orch`,
  `cuda_nvshmem_ag_gemm_host_orch`, `cuda_nvshmem_ag_gemm_seeded_orch`,
  `cuda_nvshmem_tiny_decode_host_orch`, and
  `cuda_nvshmem_tiny_decode_seeded_orch`.
- Drive both variants from the same target recipe and operator table.

### High: Runtime Loops Do Not Consume Arena Records Or Implement The Designed Common Loop

Design requirement:

- `docs/in_progress/design/overall_runtime_architecture.md:140-152` says the
  runtime loop should ask scheduler readiness, EventTensor wait completion, and
  dispatcher cursor availability, then invoke operator or sync behavior and
  report progress.
- `docs/in_progress/design/execution_model_study.md:607-636` defines the common
  task-publication contract: task records are invisible until published,
  scheduler considers published tasks, EventTensor wait blocks sync-task
  completion, and no inference is done from raw pointers.
- `docs/in_progress/design/runtime_execution_model_implementation_design.md:201-223`
  says the runtime loop invokes operators by `op_slot` from task records, not by
  example-specific task ids.

Implementation evidence:

- `include/megacu/runtime/loop/block_tile.cuh:5-28`,
  `include/megacu/runtime/loop/grid_stride.cuh:35-62`, and
  `include/megacu/runtime/loop/single_work_item.cuh:69-96` accept an `Arena`
  parameter but do not inspect it.
- `include/megacu/scheduler/explicit_asap_device.cuh:7-31` uses a fixed
  `task_count` and iterates numeric task ids; it does not read dependency attrs,
  publication state, regions, or completion state from the arena.
- `include/megacu/dispatcher/tile_grid_device.cuh:42-57` maps block ids to
  tiles but does not read per-task dispatcher attrs from arena records.

Impact:

- The new arena and execution model records are publication data, but the
  device-side execution loop does not yet use them to decide what to run.
- The current device runtime path remains closer to a manually assembled
  scheduler/dispatcher/ops wrapper than the designed arena-driven Megacu
  runtime.

Fix direction:

- Change loop/scheduler/dispatcher interfaces so the loop can iterate sealed
  arena regions and task records.
- The scheduler should evaluate explicit dependency attrs and completion state
  from published task records.
- The dispatcher should derive work cursors from the current task's
  dispatcher-owned attrs and worker context.
- The operator table should invoke by `device_task_record::op_slot`.

### High: EventTensor Device Lowering Still Depends On Manual Task Ids

Design requirement:

- `docs/in_progress/design/overall_runtime_architecture.md:79-85` says operator
  kernels do not call `notify()` or `wait()` directly; runtime loop and
  EventTensor lower attrs around operator execution or sync-only task handling.
- `docs/in_progress/design/runtime_execution_model_implementation_design.md:425-429`
  rejects examples manually assembling task ids for EventTensor notify/wait.

Implementation evidence:

- `include/megacu/backends/nvshmem/cuda_event_tensor.cuh:41-58` stores
  `notify_task` and `wait_task`, then compares `task.value` against those ids.
- `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/megacu_gemm_allreduce_phased.cu:204-210`
  manually sets `.notify_task = 0` and `.wait_task = 1`.
- `examples/cuda_nvshmem/gemm_allreduce/phased/megacu/megacu_gemm_allreduce_phased.cu:136-147`
  invokes producer on task id `0` and consumer on task id `2`.

Impact:

- EventTensor behavior is not fully owned by task attrs and the EventTensor
  component. It is still coupled to an example-specific task numbering scheme.
- This will break once host-orch/seeded-orch can produce different task ids,
  task insertion, or multiple event tensors.

Fix direction:

- EventTensor lowering should inspect `device_task_record::attributes` for
  `event_tensor::notify`, `event_tensor::wait`, and future trigger attrs.
- Sync-only wait completion should be handled by EventTensor using the current
  task record, not by hard-coded `wait_task`.
- Operator invocation should use `op_slot`, not `task.value`.

### Medium: Required Scheduler/Dispatcher Strategy Set Is Mostly Missing

Design requirement:

- `docs/in_progress/design/general_runtime_linked_components.md:109-130`
  lists required dispatcher, scheduler, execution-model, and runtime-loop
  strategies.
- `docs/in_progress/design/general_runtime_linked_components.md:667-682`
  defines selection rules for `explicit_asap`, `static_submission_order`,
  `static_level_order`, `rank_aware_tile_grid`, `tile_grid`,
  `single_work_item`, `block_tile_runtime`, `grid_stride_runtime`, and
  `single_work_item_runtime`.

Implementation evidence:

- Runtime loop structs exist for `block_tile`, `grid_stride`, and
  `single_work_item`.
- There is a host-side `run_explicit_asap` path in `src/scheduler/explicit_asap.cc`.
- There is a device-side `explicit_asap` iterator in
  `include/megacu/scheduler/explicit_asap_device.cuh`.
- No implementation of `static_submission_order`, `static_level_order`,
  `rank_aware_tile_grid`, or a real `single_work_item` dispatcher was found.
- `tests/build/runtime_strategy_selection_contracts.cc` proves loop shapes, not
  the scheduler/dispatcher strategy set described by the design.

Impact:

- The PR currently proves multiple runtime loop shapes, not full
  dispatcher/scheduler/entry strategy selection.
- The design text overstates the implemented strategy surface.

Fix direction:

- Either implement the missing strategies or explicitly downgrade them in the
  active design to future work.
- Add tests proving strategy selection changes observable scheduling or
  dispatch behavior, not only that loop types instantiate.

### Medium: Launch Adapters Are Thin Normalizers, But Not Yet Full Docker Execution Paths

Design requirement:

- `docs/in_progress/design/general_runtime_linked_components.md:80-102`
  requires direct, MPI, and torchrun launch adapters that normalize into the same
  CUDA+NVSHMEM driver.
- `docs/in_progress/design/general_runtime_linked_components.md:434-445` says
  Docker is the source of truth and must include scripts that run direct, MPI,
  and torchrun correctness paths.
- `docs/in_progress/design/general_runtime_linked_components.md:595-601` says
  adapters must be required in Docker and must not choose scheduler/dispatcher
  behavior.

Implementation evidence:

- `include/megacu/adapters/mpi_cuda_nvshmem.h:7-28` and
  `include/megacu/adapters/torch_cuda_nvshmem.h:35-56` are thin and do not
  choose scheduler/dispatcher behavior. This matches the ownership boundary.
- `tests/build/adapter_contracts.cc:150-171` validates MPI/Torch env facts into
  driver facts.
- `tools/cuda_nvshmem/run_mpi.sh:11-27` runs only
  `megacu_adapter_contracts` under `mpirun`.
- `tools/cuda_nvshmem/run_torch.py:37-67` initializes
  `torch.distributed` and runs only `megacu_adapter_contracts`.
- `tools/cuda_nvshmem/run_direct.sh:4-7` runs only tiny decode correctness.
- `tools/cuda_nvshmem/run_two_card_docker.sh:273-281` runs direct helper,
  MPI helper, Torch helper, and then old GEMM+AllReduce CTests.

Impact:

- Adapter shape is aligned, but the required Docker execution story is not.
- MPI/Torch do not launch GEMM-RS, AG-GEMM, or any Megacu distributed example.
- There is no NVSHMEM MPI/Torch bootstrap validation beyond rank facts.

Fix direction:

- Keep these adapters thin, but add real Docker correctness tests that use them
  to launch GEMM-RS and AG-GEMM Megacu binaries.
- If NVSHMEM bootstrap is still out of scope, the design should be narrowed;
  otherwise add bootstrap consistency checks and evidence.

### Medium: Tiny Decode Correctness Uses Host `phase`, Not Runtime-Owned CUDA Path

Design requirement:

- `docs/in_progress/design/general_runtime_linked_components.md:529-562`
  requires a tiny decode golden/baseline/Megacu example using the same authoring
  model as other targets.
- `docs/in_progress/design/overall_runtime_architecture.md:236-245` says all
  three Megacu variants should submit tasks, publish/seal records through the
  runtime execution model, and run through a runtime loop.
- `docs/in_progress/design/runtime_execution_model_implementation_design.md:306-328`
  requires tiny decode host-orch/seeded-orch/common variants or explicit target
  names if flat.

Implementation evidence:

- `examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_megacu.cu:279-304`
  builds a host `runtime::phase`, submits five stage functions, and calls
  `phase.run()`.
- `examples/cuda_nvshmem/tiny_decode_pipeline/README.md:50-51` states this is
  local and host-orchestrated and does not launch distributed tiny decode work.
- `examples/cuda_nvshmem/tiny_decode_pipeline/README.md:63-65` says the current
  correctness test does not run CUDA kernels or distributed work.

Impact:

- Tiny decode is useful as a host-side authoring-surface correctness test, but
  it does not validate the runtime-owned device loop, `host-orch`, or
  `seeded-orch`.

Fix direction:

- Keep the current host-phase path as a small correctness oracle if useful, but
  add Megacu host-orch and seeded-orch variants that use task arena records and
  runtime loops.

### Medium: Stable `docs/design/` Was Edited Despite Active Design Saying To Recreate It At Closeout

Design/process requirement:

- `docs/in_progress/design/stable_docs_recreation_plan.md:3-4` says not to
  apply the stable docs recreation directly to `docs/design/` until PR merge.
- `docs/in_progress/design/stable_docs_recreation_plan.md:145-166` says to
  remove and recreate the old stable folder during PR closeout.

Implementation evidence:

- `git diff --name-only main..HEAD -- docs/design docs/in_progress/design`
  shows edits to eight files under
  `docs/design/runtime_linked_device_native_layer/`.

Impact:

- This creates a lifecycle mismatch: active design says stable docs should wait
  for closeout, but the branch has already edited stable PR #4 docs.
- The edits appear mostly to remove rejected terminology and stale wording, but
  they still conflict with the documented closeout plan.

Fix direction:

- Either revert direct `docs/design/` edits and keep the rejected-note handling
  solely in `docs/in_progress/design/stable_docs_recreation_plan.md`, or
  explicitly decide that these narrow stale-term cleanups are allowed before
  full closeout.

### Low: Contract Tests Use `assert`, So Some Failures Vanish Under `NDEBUG`

Implementation evidence:

- Several build/runtime contracts use `assert`, for example
  `tests/build/runtime_execution_models_contracts.cc` and
  `tests/build/runtime_device_persistent_contract.cu`.
- `tests/runtime/tiny_decode_correctness.cc` uses explicit return codes, which
  is safer for release builds.

Impact:

- If these tests are built with `NDEBUG`, some contract checks may compile and
  pass without checking behavior.

Fix direction:

- Convert new contract tests that are intended as behavioral evidence to
  explicit failure returns or a small local expectation helper.

## Design Correspondence Matrix

| Design area | Current code status | Correspondence |
| --- | --- | --- |
| Thin raw operator arguments | `runtime::phase::submit` accepts native function signatures and raw args in `include/megacu/runtime.h:318-351`; tiny decode submits raw pointer args in `tiny_decode_megacu.cu:285-301`. | Mostly aligned for host surface. Device arena path does not yet carry raw args. |
| Explicit dependencies only | `scheduler::depends_on` exists in `include/megacu/runtime.h:94-100`; host scheduler checks dependency attrs in `src/scheduler/explicit_asap.cc:40-52`. | Partially aligned. No inference, but device scheduler does not read task deps from arena. |
| EventTensor API naming | `event_tensor::shape/wait_count/notify/wait/trigger` exist in `include/megacu/runtime.h:102-126`; stale `runtime::events` guard exists in CMake. | Aligned at API naming level. Lowering is still task-id coupled. |
| EventTensor as sync-task completion condition | Host surface places wait on sync task in `gemm_allreduce_orchestrate_common.h:52-54`; tests check consumer does not carry wait. | Aligned in host records. Device loop does not yet evaluate wait attrs from records. |
| Runtime, not entry | `device_entry.cuh` was removed; `block_tile_runtime.cuh` delegates to `runtime::loop::block_tile`; CMake guard rejects old entry naming. | Mostly aligned. |
| Runtime owns execution model and loop | `device_persistent` composes execution and loop in `include/megacu/runtime/device_persistent.cuh:22-39`. | Aligned structurally. Not integrated into examples. |
| Host-orch | `host_orch::frame` exists and records tasks/events/deps/region. | Partial. Contract test only; no example. |
| Seeded-orch | `seeded_orch::model` and `device_orch` exist and publish/seal on device. | Partial. Contract test only; no example; no distributed validation. |
| Runtime loops | `block_tile`, `grid_stride`, `single_work_item` loops exist. | Partial. Loops do not use arena records or task attrs. |
| Dispatcher candidates | `tile_grid_device` exists; host `map_explicit_attrs` counts attrs. | Partial. Missing `rank_aware_tile_grid` and real `single_work_item` dispatcher. |
| Scheduler candidates | Host and device `explicit_asap` exist. | Partial. Missing `static_submission_order` and `static_level_order`; device ASAP ignores deps. |
| Launch adapters | MPI/Torch adapter headers and env contracts exist. | Partial/aligned for thin normalization; missing real example launch correctness and NVSHMEM bootstrap proof. |
| GEMM-RS example | Directory and skeleton object target exist. | Missing required end-to-end behavior. |
| AG-GEMM example | Directory and skeleton object target exist. | Missing required end-to-end behavior. |
| Tiny decode example | Golden/baseline/Megacu host-phase correctness exists. | Partial. Not runtime-owned CUDA path, not host-orch/seeded-orch. |
| Docker source of truth | Docker installs MPI/Torch and script runner invokes helpers. | Partial. Helpers do not run required GEMM-RS/AG-GEMM correctness. |
| Stable docs lifecycle | Closeout plan exists under in-progress. | Partially misaligned because `docs/design/` also changed. |

## Positive Notes

- The adapter headers are appropriately thin: they normalize launch facts into
  `driver_view` and do not select scheduler, dispatcher, runtime, or operator
  behavior.
- The public EventTensor naming now follows the accepted terminology.
- `submit` and `sync` keep operator tasks and sync-only tasks separate at the
  host surface.
- `device_persistent` is the right structural home for execution model plus
  runtime loop.
- The seeded-orch tests check several important failure modes: task overflow,
  dependency overflow, event overflow, region overflow, and failure gating before
  the loop runs.
- The CMake grep guards are useful for preventing regression to user-programmed
  `scheduler.run`, stale event API names, and old `entry` naming.

## Recommended Fix Order

1. Make the device runtime loop arena-driven.
   - Loop over sealed regions and task records.
   - Scheduler readiness should use explicit dependency attrs and completion
     state.
   - EventTensor should evaluate notify/wait attrs from the current task record.
   - Dispatcher should return cursors based on current task attrs and worker
     context.

2. Remove task-id coupling from EventTensor and operator invocation.
   - Replace `.notify_task = 0`, `.wait_task = 1`, and task-value branches with
     `op_slot` and task attrs.

3. Integrate `host-orch` and `seeded-orch` into one small example first.
   - Tiny decode is the lowest-risk target because it already has golden and
     baseline correctness.
   - Add host-orch and seeded-orch target names and tests.

4. Implement one distributed example end to end before adding the second.
   - Prefer GEMM-RS first because the design uses it as the adapter proof.
   - Add golden, baseline, Megacu, direct, MPI, and Torch paths.

5. Add AG-GEMM using the same architecture and test shape.

6. Decide the strategy set honestly.
   - Either implement `rank_aware_tile_grid`, `static_submission_order`, and
     `static_level_order`, or move them to future work in the active design.

7. Resolve docs lifecycle before PR publication.
   - Either revert direct `docs/design/` edits or explicitly bless them as
     narrow stale-term cleanup.
   - The full stable docs recreation still belongs at PR closeout.

## Open Questions For The Human Review

1. Should this PR remain scoped to the full active design, or should it be split
   into a smaller runtime-contract PR plus a later distributed-example PR?
2. Is the current host `runtime::phase` path still allowed as a user-facing
   convenience, or should all Megacu examples now route through
   `host-orch`/`seeded-orch` only?
3. Should the first arena-driven device scheduler be a simple static
   dependency scan, or should it include a ready-but-not-issued buffer now?
4. For MPI/Torch adapters, is rank/local-rank normalization enough for this PR,
   or must the adapter also prove NVSHMEM bootstrap consistency?
5. Should stable `docs/design/` edits be reverted until PR closeout?

## Residual Risk

- GPU, CUDA, NVSHMEM, MPI, and Torch distributed execution were not exercised in
  this review pass.
- The current tests may still pass while the active design remains unmet because
  many checks validate shape, layout, or contract compilation rather than
  distributed runtime behavior.
- The existing GEMM+AllReduce proof remains useful history, but it is explicitly
  not the target example scope for this PR.
- The branch is clean and organized into commits, but commit organization does
  not imply architecture completeness.
