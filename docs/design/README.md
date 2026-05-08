# Design

`docs/design/` contains implemented behavior only.

During feature work, draft designs live under `docs/in_progress/design/`.
Before a task closes, validated design content should move here and stale
in-progress drafts should be removed.

## Implemented Documents

- `docs/design/agent_harness.md` - repo-local agent harness, docs lifecycle,
  source-reading policy, GitHub workflow skills, and human-agent workflow.
- `docs/design/megacu_cpp_cuda_layer.md` - picked Megacu device-native layer
  direction and current stable boundaries.
- `docs/design/runtime_architecture.md` - implemented runtime-owned execution
  model, loop, scheduler, dispatcher, EventTensor, platform, backend, and
  driver boundaries.
- `docs/design/event_tensor.md` - implemented Megacu EventTensor semantics,
  task-level synchronization, and comparison with the Event Tensor paper.
- `docs/design/program_compile_execute_flow.md` - intuitive end-to-end flow
  from operator tasks to one CUDA+NVSHMEM mega-kernel.
- `docs/design/cuda_nvshmem_examples.md` - implemented GEMM-RS, AG-GEMM, and
  tiny decode example contracts.
- `docs/design/launch_adapters_and_verification.md` - direct, MPI, Torch, and
  Docker verification paths.
