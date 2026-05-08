# General Runtime Examples Source Notes

- Date: 2026-05-04 Asia/Shanghai
- Purpose: identify required example/design inputs for the general
  runtime-linked components PR.
- Related design:
  `docs/design/runtime_architecture.md`,
  `docs/design/cuda_nvshmem_examples.md`, and
  `docs/design/launch_adapters_and_verification.md`

## Sources Read

### Triton-distributed

- Repository: `research/repos/triton-distributed`
- Relevant files:
  - `tutorials/07-...-allgather-gemm.py` (AG-GEMM tutorial)
  - `tutorials/08-...-gemm-reduce-scatter.py` (GEMM-RS tutorial)
  - `python/triton_dist/kernels/nvidia/gemm_allreduce.py`
  - `python/triton_dist/test/nvidia/test_gemm_ar.py`
  - `python/triton_dist/test/nvidia/test_tp_e2e.py`
  - `python/triton_dist/test/nvidia/test_e2e_inference.py`
  - `python/triton_dist/utils.py`
  - `python/little_kernel/design/test_flashcomm_torchrun.py`

Design lessons for this PR:

- Distributed launch should be a first-class verification path, not a
  best-effort optional feature. Triton-distributed exercises tensor-parallel
  and distributed GEMM/communication paths through launched multi-rank tests.
- `torchrun` is used as a control-plane launcher: ranks, local ranks, and world
  size are derived from environment/process-group state before the backend
  initializes communication.
- NVSHMEM unique-id bootstrap through a torch process group is the right
  pattern for a Torch adapter that does not depend on MPI.
- Tutorial 08's GEMM-RS shape is a strong distributed-kernel validation
  pattern: GEMM tile producer work publishes readiness for reduce-scatter
  consumer work, while rank-aware tile ordering and per-destination readiness
  matter.
- Tutorial 07's AG-GEMM shape is the complementary validation pattern:
  all-gather producer work fills peer-visible regions and GEMM tile consumer
  work waits for those regions to become ready.
- Together, GEMM-RS and AG-GEMM force the runtime to express rank identity,
  symmetric storage, tile readiness, backend primitive selection, and
  dispatcher/scheduler work partitioning without embedding the algorithm in one
  handwritten Megacu kernel.
- The test shape should include correctness against a baseline/golden path and
  should run under Docker with the same dependencies the PR claims.

Megacu implications:

- Add required Docker tests for direct, MPI, and torchrun launch paths.
- The Torch adapter should only normalize launch/control-plane state and
  initialize the backend; it must not decide task placement or scheduling.
- The MPI adapter should similarly normalize MPI rank/local-rank state and
  backend bootstrap into the same `cuda_nvshmem::driver`.
- More dispatcher/scheduler/runtime strategies should be exercised with real
  targets, not only compile anchors.
- The Megacu variants of GEMM-RS and AG-GEMM should be composed from
  tile-operator tasks, sync-only EventTensor tasks, and explicit dependency
  attrs. Handwritten large fused/persistent kernels are baselines only.

### HazyResearch Megakernels / MegaKittens

- Repository: `research/repos/hazy-megakernels-mk-v2-llama-70b`
- Relevant files:
  - `examples/llama1b/compiled_decode.py`
  - `examples/llama1b/scheduler.py`
  - `examples/llama1b/benchmark_instructions.py`
  - `tests/test_mlp.py`
  - `benchmarks/benchmark_mlp.py`
  - `csrc/itypes/llama1b/*.cuh`
  - `csrc/itypes/reference/*.cu`

Design lessons for this PR:

- The useful end-to-end pattern is a decode pipeline, not an isolated numeric
  kernel. Hazy's Llama1B example schedules a sequence of instruction-like
  operator bodies: normalization, QKV/rope/cache update, attention,
  projection/residual, MLP, and LM head.
- The hand scheduler names tensor slots, instruction records, barrier indices,
  source barriers, destination barriers, and per-layer sequencing explicitly.
- The compiled-decode example proves the same high-level computation can be
  represented as operator invocations and scheduled into a mega-kernel-shaped
  execution path.
- Smaller MLP tests and benchmarks show the baseline/golden comparison pattern:
  run a plain reference function, run the mega-kernel path, compare output, and
  optionally time both.

Megacu implications:

- Add a runnable Hazy-style end-to-end example, scaled down to avoid external
  model downloads and full Llama1B memory requirements.
- The example should be a tiny decode pipeline with multiple operator kinds and
  explicit readiness: e.g. RMS/norm-like transform, two matvec/MLP stages,
  residual/activation, and final logits.
- Keep three roles separate:
  - **golden**: plain CPU or straightforward CUDA reference for expected
    numeric output;
  - **baseline**: handwritten CUDA mega-kernel or conventional native sequence;
  - **Megacu**: orchestrate submits operator tasks and runs through linked
    dispatcher/scheduler/runtime strategies.
- The example should run in Docker and compare Megacu against both baseline and
  golden. Performance comparison can be reported as timing only if measured;
  correctness is mandatory.

### Simpler / PTO Runtime

- Repository: `research/repos/simpler`
- Relevant files:
  - `docs/orchestrator.md`
  - `docs/scheduler.md`
  - `docs/task-flow.md`
  - `src/a2a3/runtime/host_build_graph/docs/RUNTIME_LOGIC.md`
  - `src/a2a3/runtime/aicpu_build_graph/docs/RUNTIME_LOGIC.md`
  - `src/a2a3/runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md`

Design lessons for this PR:

- Simpler has a clean split between an orchestrator that submits task slots,
  a scheduler that moves slots through wiring/ready/completion queues, and
  workers that execute opaque callables with task arguments and call config.
- It derives dependencies from tensor input/output/inout-style tags through a
  TensorMap. That is the exact dependency path Megacu should not use.
- Simpler also keeps separate runtime variants for different build/run models:
  host-built graph, device-side graph construction, and ring-buffer based
  device orchestration. Megacu should adopt this flexibility at the architecture
  level rather than hard-coding one host-built execution model.

Megacu implications:

- Keep task handles and call-owned task records, but do not adopt TensorMap
  dependency inference.
- Keep explicit attrs as the only dependency/event source.
- Add build-run model as a distinct strategy: host-built snapshot,
  device-orch persistent, hybrid seeded device-orch, and future host-streaming
  can share scheduler/dispatcher/EventTensor contracts while differing in task
  publication and lifetime.
- Use Simpler as a contrast point for why `scheduler.run` should not be a user
  API and why the runtime owns the execution model and internal device loop.

## Architecture Comparison Summary

| System | Useful lesson | Boundary Megacu should not copy |
| --- | --- | --- |
| MegaKittens/Hazy | Explicit instruction/barrier readiness and end-to-end decode pipeline shape. | Fixed instruction ABI, fixed worker roles, model-specific tensor slots. |
| Simpler/PTO Runtime | Clear orchestrator/scheduler/worker split and task handle flow. | TensorMap dependency inference and host queue execution as Megacu semantics. |
| Triton-distributed | Torch/NVSHMEM launch facts, symmetric storage, rank-aware tests, device communication protocols. | Embedding all scheduling and communication protocol choices inside one hand-written kernel. |

The Megacu design should keep explicit task/event objects, a linked device
runtime ownership, and narrow dispatcher/scheduler/event/platform/backend
components. Runtime replaces the earlier `entry` name because it owns the
execution model, device-side loop, and issue policy.

## Selected Example Shape

The PR should implement three example families:

```text
examples/cuda_nvshmem/gemm_reduce_scatter/
examples/cuda_nvshmem/allgather_gemm/
examples/cuda_nvshmem/tiny_decode_pipeline/
```

The first two are Triton-distributed-inspired distributed examples. GEMM-RS
proves GEMM producer tiles feeding reduce-scatter consumer work. AG-GEMM proves
all-gather producer work feeding GEMM consumer tiles. The third is Hazy-style
because it is an end-to-end decode pipeline with multiple
operator stages, explicit task/event readiness, a reusable schedule/runtime
strategy, and baseline/golden comparisons. It is intentionally not a full
Llama1B/HuggingFace example because the PR needs deterministic Docker
verification without model downloads or large-memory assumptions.

Expected variants:

```text
examples/cuda_nvshmem/tiny_decode_pipeline/
  common/
  golden/
  baseline/
  megacu/
```

The same variant shape should be used for GEMM-RS and AG-GEMM. The distributed
examples should support one host / one GPU and one host / two GPU runs through
direct, MPI, and Torch launch paths. The tiny decode example should support one
host and at least one GPU; distributed tiny decode can remain future work if
GEMM-RS and AG-GEMM already prove the distributed adapters in this PR.

## Verification Evidence To Add

- Docker direct tiny decode correctness: golden, baseline, Megacu.
- Docker direct/MPI/torchrun GEMM-RS correctness.
- Docker direct/MPI/torchrun AG-GEMM correctness.
- Strategy-selection tests proving the tiny decode example can use a different
  dispatcher/scheduler/runtime combination from the distributed tiled examples.
