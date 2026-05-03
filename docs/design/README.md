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
- `docs/design/runtime_linked_device_native_layer/` - implemented
  runtime-linked dispatcher/scheduler/platform/backend architecture and the
  CUDA+NVSHMEM GEMM+AllReduce proof.
