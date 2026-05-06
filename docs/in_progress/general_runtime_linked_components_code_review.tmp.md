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
| Task/Event records | Compact records for host-orch and seeded-orch execution | Arena and record types exist and the first runtime arena execution checkpoint consumes sealed task/event records | Mostly complete | `task_arena` contracts and runtime arena execution contracts |
| `host-orch` model | Host builds task records, runtime launches a snapshot | Frame contracts exist and tiny decode now uses a host-built arena in a CUDA megakernel; distributed examples remain incomplete | Partial | Host-orch contracts plus `tiny_decode_correctness` |
| `seeded-orch` model | Host seeds device arena; persistent runtime can build/consume work dynamically | Device publication contracts exist and tiny decode now uses device-side recipe publication in a CUDA megakernel; distributed examples remain incomplete | Partial | Seeded-orch contracts plus `tiny_decode_correctness` |
| Runtime composition | Link scheduler, dispatcher, EventTensor, and operators behind a common runtime API | `device_persistent<ExecutionModel, Loop, ...>` provides a composition point | Mostly complete | Compile contracts for loop candidates |
| Runtime loops | Overrideable internal loops owned by runtime | `block_tile` now drives arena execution smoke; other loop candidates remain compile/runtime contracts | Mostly complete for arena execution smoke | Runtime arena execution CUDA smoke |
| Scheduler, host side | ASAP readiness from explicit dependencies only | `explicit_asap` exists as a host-side contract | Partial | Contract tests |
| Scheduler, device side | Common task readiness for runtime loops | Device ASAP reads sealed arena records and explicit deps in the smoke path | Mostly complete for explicit deps over sealed arena records | Runtime arena execution CUDA smoke |
| Dispatcher, host side | Decode dispatch attrs into work cursor/placement decisions | Attr counter/contract path exists | Partial | Dispatcher tests |
| Dispatcher, device side | Pick spatial work for selected ready task | `tile_grid` derives first/next work from task attrs in the smoke path; broader dispatchers remain missing | Mostly complete for tile-grid first/next from attrs | Runtime arena execution CUDA smoke |
| EventTensor API | Event shape plus notify/wait/trigger attrs used by sync tasks | API and attr contracts exist, with CUDA attr-lowering smoke | Mostly complete with CUDA attr-lowering smoke | EventTensor tests, docs, and runtime arena execution CUDA smoke |
| CUDA+NVSHMEM EventTensor lowering | Backend/platform implementation for sync completion | Attr-driven smoke adapter exists; mapping is still smoke-level, not full distributed EventTensor mapping | Partial, attr-driven smoke added | Runtime arena execution CUDA smoke; GEMM+AllReduce proof path remains legacy |
| Driver | Carry CUDA+NVSHMEM launch/rank/team facts into runtime | Short driver view exists; conceptually acceptable, but more distributed fields may be needed as examples become real | Partial | CUDA+NVSHMEM driver/adapters |
| MPI adapter | Required distributed launch path | Normalizes MPI facts into driver facts | Partial | Adapter contracts; no full example run evidence |
| Torch adapter | Required distributed launch path | Normalizes torchrun environment into driver facts | Partial | Adapter contracts; no full example run evidence |
| Docker environment | Required execution envelope for MPI/Torch/CUDA+NVSHMEM | Helper scripts exist, but do not prove all required examples | Partial | Docker/tooling findings below |
| GEMM-RS example | Required end-to-end distributed example | Golden, baseline, Megacu host-orch, and Megacu seeded-orch local CUDA correctness exists | Partial | Distributed direct/MPI/Torch correctness still missing |
| AG-GEMM example | Required end-to-end distributed example | Directory/build skeleton exists | Missing | No golden/baseline/Megacu correctness binary |
| Tiny decode example | Required end-to-end decode pipeline example | Golden/baseline/Megacu host-orch and seeded-orch local CUDA correctness path exists | Partial | Distributed tiny decode path still missing |
| Stable docs closeout | Recreate flattened `docs/design/` at merge time | Active in-progress docs record this requirement | Process/deferred | Stable docs were touched early; see finding |

Task completion status:

| Task/design area | Target requirement | Current proof | Status |
| --- | --- | --- | --- |
| Architecture documentation | Active design explains runtime, scheduler, dispatcher, EventTensor, adapters, examples, and closeout docs | `docs/in_progress/design/` contains the discussion and implementation spec | Mostly complete |
| Runtime/EventTensor API rename and decoupling | Use `runtime`, not `entry`; use `EventTensor`, not EventEngine; keep EventTensor as task attrs, not operator-manual protocol | Headers/docs/tests reflect the rename | Mostly complete |
| Explicit task dependency model | Dependencies are explicit attrs and scheduler-owned | Host contracts and CUDA arena smoke read explicit deps | Mostly complete for this checkpoint |
| Runtime-owned internal loop | Runtime owns loop strategies, not scheduler or examples | Loop strategy candidates compile; `block_tile` consumes arena records in CUDA smoke | Mostly complete for arena execution smoke |
| Host-orch execution model | Snapshot-style host built task plan | Shared recipe topology contract exists | Partial |
| Seeded-orch execution model | Device-resident orchestration seeded from host | Shared recipe equivalence CUDA smoke exists | Partial |
| Scheduler/dispatcher strategy set | Provide multiple strategies that can be picked through the common runtime API | Some runtime loops exist; scheduler/dispatcher candidates are thin | Partial |
| MPI/Torch required adapters | Required by PR and Docker path, not optional dependency paths | Adapter normalization exists | Partial |
| GEMM-RS | Tile-operator Megacu example with golden and baseline comparison | Local CUDA correctness exists | Partial |
| AG-GEMM | Tile-operator Megacu example with golden and baseline comparison | Skeleton only | Missing |
| Tiny decode | Pipeline example with golden, baseline, and Megacu comparison | Local CUDA host-orch and seeded-orch paths exist | Partial |
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

Skipped during the original review:

- No build or CTest was rerun as part of this review pass.
- No Docker execution was run.
- No GitHub state was changed.

The previous implementation pass reported a local targeted build and CTest
result of 18/18 passing, but this review treats Docker and full distributed
correctness evidence as missing unless implemented and documented in code.

Runtime arena checkpoint evidence reported by task implementers/reviewers during
Tasks 1-5:

- `cmake --build build --target megacu_runtime_arena_execution_contracts`
- `ctest --test-dir build -R runtime_arena_execution_contracts --output-on-failure`
- `cmake --build build --target megacu_runtime_arena_execution_contract`
- `ctest --test-dir build -R runtime_arena_execution_contract --output-on-failure`
- `cmake --build build --target megacu_runtime_device_persistent_contract megacu_runtime_arena_execution_contracts`
- `ctest --test-dir build -R 'runtime_device_persistent_contract|runtime_arena_execution_contracts' --output-on-failure`

These targeted checks were run by the task implementers during Tasks 1-5. Full
Docker/NVSHMEM distributed examples were not run.

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
- First runtime arena execution checkpoint: host-only topology contract plus CUDA
  smoke proving shared recipe records, explicit device deps, tile-grid dispatch
  attrs, sync-only EventTensor wait, operator notify, completion state, and
  seeded-orch shared recipe equivalence.
- Thin MPI/Torch launch fact adapters that normalize rank/world/local-rank into
  `cuda_nvshmem::driver_view`.
- Tiny decode golden/baseline/Megacu host-orch and seeded-orch local CUDA
  correctness path.
- GEMM-RS local CUDA correctness path and AG-GEMM buildable skeleton object
  target.

What is not implemented relative to the active design:

- AG-GEMM is not an end-to-end example; it is still a skeleton.
- No AG-GEMM golden, baseline, or Megacu correctness binary exists.
- GEMM-RS has one-host/one-GPU correctness, but no direct two-GPU, MPI, or
  Torch distributed correctness path exists yet.
- No direct/distributed AG-GEMM correctness path exists.
- `host-orch` and `seeded-orch` are integrated into tiny decode and
  GEMM-AllReduce, but not into GEMM-RS or AG-GEMM.
- Runtime arena smoke is not yet the GEMM-AllReduce refit or an end-to-end
  distributed example.
- EventTensor mapping is still smoke-level, not a general multi-tile distributed
  mapping.
- Required dispatcher/scheduler strategy candidates are not all implemented.
- Docker helper scripts exercise adapter contracts and old GEMM+AllReduce tests,
  not the required new PR examples.

Merge-readiness assessment:

- Ready as a partial internal contract/prototype PR only if the PR scope is
  explicitly narrowed.
- Not ready for the stated active PR goal in
  `docs/in_progress/design/general_runtime_linked_components.md`.

## Findings

### High: AG-GEMM Is Still A Skeleton And GEMM-RS Is Local-Only

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

- `examples/cuda_nvshmem/gemm_reduce_scatter/` now contains common, golden,
  baseline, and Megacu host-orch/seeded-orch local CUDA correctness paths.
- `examples/cuda_nvshmem/allgather_gemm/megacu/allgather_gemm_megacu.cu:1-3`
  contains only a skeleton function returning `1`.
- `examples/cuda_nvshmem/gemm_reduce_scatter/README.md` documents that the
  distributed numeric GEMM-RS run remains pending.
- `examples/cuda_nvshmem/allgather_gemm/README.md:134-141` says the same for
  AG-GEMM.

Impact:

- The current branch proves the GEMM-RS task composition locally, but not yet
  through distributed direct/MPI/Torch launch paths.
- AG-GEMM still does not exercise scheduler, dispatcher, EventTensor, runtime
  execution model, or backend behavior.

Fix direction:

- Implement `common/`, `golden/`, `baseline/`, and `megacu/` for AG-GEMM.
- Add GEMM-RS distributed correctness binaries for direct, MPI, and Torch
  launch paths.
- The Megacu variants should submit producer tasks, sync-only EventTensor wait
  tasks, and consumer tasks with explicit dependency attrs.

### High: `host-orch` And `seeded-orch` Are Not In Every Required Example

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
- `examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_megacu.cu`
  now exposes host-orch and seeded-orch CUDA megakernel variants from one shared
  recipe.
- AG-GEMM Megacu source is still a skeleton. GEMM-RS now has local CUDA
  host-orch and seeded-orch variants.

Impact:

- The core ownership claim is now demonstrated by tiny decode, GEMM-AllReduce,
  and local GEMM-RS, but still not by AG-GEMM.

Fix direction:

- Add AG-GEMM host-orch and seeded-orch variants from one shared target recipe
  and operator table.
- Add distributed validation for GEMM-RS and tiny decode.

### Medium: Runtime Arena Smoke Is Landed, But Example Coverage Is Incomplete

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

- `include/megacu/runtime/loop/block_tile.cuh` now consumes sealed arena task
  records in the runtime arena execution smoke.
- `include/megacu/scheduler/explicit_asap_device.cuh` now reads arena deps and
  task completion state for explicit ASAP readiness.
- `include/megacu/dispatcher/tile_grid_device.cuh` now derives tile-grid work
  from task attrs for the smoke contract.
- `tests/build/runtime_arena_execution_contract.cu` proves the three-task
  recipe executes producer, sync-only wait, and consumer work through the
  composed runtime.
- `grid_stride` and `single_work_item` remain candidates that are not proven on
  the arena execution contract.

Impact:

- The common-loop checkpoint is real and now has GEMM-AllReduce, GEMM-RS local,
  and tiny decode local example coverage, but AG-GEMM and distributed variants
  remain incomplete.
- The CUDA contract is monolithic and should be split as the example contracts
  harden.

Fix direction:

- Refit AG-GEMM through the same arena-driven runtime loop.
- Split the monolithic CUDA contract into focused scheduler, dispatcher,
  EventTensor, loop, and execution-model contracts once the interfaces harden.
- Extend or replace the remaining loop candidates with arena-backed behavior
  before claiming loop-strategy completeness.

### Medium: EventTensor Lowering Is Attr-Driven Only At Smoke Level

Design requirement:

- `docs/in_progress/design/overall_runtime_architecture.md:79-85` says operator
  kernels do not call `notify()` or `wait()` directly; runtime loop and
  EventTensor lower attrs around operator execution or sync-only task handling.
- `docs/in_progress/design/runtime_execution_model_implementation_design.md:425-429`
  rejects examples manually assembling task ids for EventTensor notify/wait.

Implementation evidence:

- `include/megacu/backends/nvshmem/cuda_event_tensor.cuh` now includes an
  attr-driven smoke adapter that scans task attrs for EventTensor notify/wait.
- Notify lowering uses the operator work tile, and sync-task wait lowering
  covers the EventTensor shape recorded on the event object.
- The older task-id-coupled proof type remains present for compatibility with
  handwritten baseline history only; the active GEMM-AllReduce Megacu path uses
  `attr_event_tensor_i32`.
- `examples/cuda_nvshmem/gemm_allreduce/megacu/megacu_gemm_allreduce.cu`
  dispatches operators by `op_slot` from the arena task record.

Impact:

- EventTensor behavior is now proven through attrs for the runtime arena smoke
  and the active GEMM-AllReduce direct correctness path. Full distributed
  NVSHMEM validation is still pending.

Fix direction:

- Complete the distributed 1-host-2-device validation path through direct,
  MPI, and Torch launch adapters.

- Generalize the attr-driven adapter beyond the smoke mapping.
- Refit GEMM-AllReduce and later distributed examples so they no longer pass
  `.notify_task`/`.wait_task`.
- Add distributed EventTensor mapping checks before claiming CUDA+NVSHMEM
  lowering is complete.

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

- `examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_runtime_recipe.cuh`
  defines one shared recipe with operator tasks, sync-only EventTensor waits,
  explicit `depends_on` attrs, and tile-grid dispatch attrs.
- `examples/cuda_nvshmem/tiny_decode_pipeline/megacu/tiny_decode_megacu.cu`
  launches the recipe through both host-orch and seeded-orch runtime execution
  models in CUDA megakernels.
- `tests/runtime/tiny_decode_correctness.cc` compares golden, baseline,
  Megacu host-orch, Megacu seeded-orch, and the compatibility Megacu ABI.

Impact:

- Tiny decode now validates the runtime-owned device loop, `host-orch`, and
  `seeded-orch` locally. It still does not validate distributed tiny decode.

Fix direction:

- Add the distributed 1-host-2-device tiny decode path when the PR reaches the
  distributed example validation slice.

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
| Thin raw operator arguments | `runtime::phase::submit` accepts native function signatures and raw args in `include/megacu/runtime.h:318-351`; runtime-linked examples pass raw pointers through target runtime args and operator tables. | Mostly aligned for the current examples. |
| Explicit dependencies only | `scheduler::depends_on` exists in `include/megacu/runtime.h:94-100`; host scheduler checks dependency attrs in `src/scheduler/explicit_asap.cc:40-52`; CUDA smoke checks device arena deps. | Mostly aligned for the checkpoint. Missing full example validation. |
| EventTensor API naming | `event_tensor::shape/wait_count/notify/wait/trigger` exist in `include/megacu/runtime.h:102-126`; stale `runtime::events` guard exists in CMake. | Aligned at API naming level. CUDA attr-lowering smoke exists; distributed mapping remains partial. |
| EventTensor as sync-task completion condition | The shared GEMM-AllReduce recipe places wait on a sync task; CUDA arena smoke and GEMM-AllReduce direct correctness complete sync-only waits through EventTensor attrs. | Mostly aligned for the checkpoint. Missing distributed example validation. |
| Runtime, not entry | `device_entry.cuh` was removed; `block_tile_runtime.cuh` delegates to `runtime::loop::block_tile`; CMake guard rejects old entry naming. | Mostly aligned. |
| Runtime owns execution model and loop | `device_persistent` composes execution and loop in `include/megacu/runtime/device_persistent.cuh:22-39`; GEMM-AllReduce host-orch and seeded-orch use this path. | Mostly aligned for the active tracer bullet. |
| Host-orch | `host_orch::frame` records tasks/events/deps/region; GEMM-AllReduce direct correctness uses the host-orch runtime variant. | Partial. Direct 1-host-1-device proven; distributed validation pending. |
| Seeded-orch | `seeded_orch::model` and `device_orch` publish/seal on device; GEMM-AllReduce direct correctness uses the seeded-orch runtime variant. | Partial. Direct 1-host-1-device proven; distributed validation pending. |
| Runtime loops | `block_tile`, `grid_stride`, `single_work_item` loops exist; `block_tile` executes arena records in smoke. | Mostly aligned for first arena checkpoint. Other loops and examples remain partial. |
| Dispatcher candidates | `tile_grid_device` exists and reads task attrs in smoke; host `map_explicit_attrs` counts attrs. | Partial. Missing `rank_aware_tile_grid` and real `single_work_item` dispatcher. |
| Scheduler candidates | Host and device `explicit_asap` exist; device ASAP reads sealed arena deps in smoke. | Partial. Missing `static_submission_order` and `static_level_order`; no full example validation. |
| Launch adapters | MPI/Torch adapter headers and env contracts exist. | Partial/aligned for thin normalization; missing real example launch correctness and NVSHMEM bootstrap proof. |
| GEMM-RS example | Golden/baseline/Megacu host-orch and seeded-orch local CUDA correctness exists. | Partial. Distributed launch correctness remains missing. |
| AG-GEMM example | Directory and skeleton object target exist. | Missing required end-to-end behavior. |
| Tiny decode example | Golden/baseline/Megacu host-orch and seeded-orch local CUDA correctness exists. | Partial. Distributed tiny decode is still missing. |
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
- The runtime arena execution checkpoint now proves a shared three-task recipe
  through host-orch topology and seeded-orch CUDA runtime execution smoke.
- The seeded-orch tests check several important failure modes: task overflow,
  dependency overflow, event overflow, region overflow, and failure gating before
  the loop runs.
- The CMake grep guards are useful for preventing regression to user-programmed
  `scheduler.run`, stale event API names, and old `entry` naming.

## Recommended Fix Order

1. Refit GEMM-AllReduce through the arena-driven runtime checkpoint.
   - Keep the shared recipe shape: operator task, sync-only EventTensor wait,
     consumer task, explicit deps, and runtime-owned loop.
   - Replace legacy `.notify_task`/`.wait_task` example wiring with EventTensor
     attrs.

2. Split the monolithic CUDA runtime arena contract as interfaces harden.
   - Preserve the current end-to-end smoke, but add focused contracts for
     scheduler, dispatcher, EventTensor, runtime loop, and seeded-orch
     equivalence.

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
2. Should the first arena-driven device scheduler be a simple static
   dependency scan, or should it include a ready-but-not-issued buffer now?
3. For MPI/Torch adapters, is rank/local-rank normalization enough for this PR,
   or must the adapter also prove NVSHMEM bootstrap consistency?
4. Should stable `docs/design/` edits be reverted until PR closeout?

## Residual Risk

- GPU, CUDA, NVSHMEM, MPI, and Torch distributed execution were not exercised in
  this review pass.
- Full direct/MPI/Torch/Docker example matrix remains missing.
- EventTensor mapping is still smoke-level: `event ref -> tile` and
  `event ref + 1 -> ready value`, not general multi-tile distributed mapping.
- Runtime arena smoke is not yet the GEMM-AllReduce refit.
- The CUDA contract is monolithic and should later be split as examples harden.
- The current tests may still pass while the active design remains unmet because
  many checks validate shape, layout, or contract compilation rather than
  distributed runtime behavior.
- The existing GEMM+AllReduce proof remains useful history, but it is explicitly
  not the target example scope for this PR.
- The branch is clean and organized into commits, but commit organization does
  not imply architecture completeness.
