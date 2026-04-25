# Megacu

Megacu is an early-stage project for a thin device-native C++ layer for
building megakernels with CUDA as the first platform and performance baseline.
The core performance goal is zero abstraction overhead on the hot path: users
should write normal CUDA/CuTe/CUTLASS/NVSHMEM device code, while Megacu
supplies explicit task, event, and scheduling building blocks.

The first project invariant is process quality: agent-friendly workflow,
recorded source readings, explicit design, and verification evidence come
before implementation.

Start with:

- `AGENTS.md` for agent operating rules
- `docs/design/megacu_cpp_cuda_layer.md` for the accepted device-native layer
  design
- `docs/in_progress/design/implementation_ready_device_native_layer/` for the
  active first-implementation redesign
- `docs/design/agent_harness.md` for the repo-local collaboration harness
- `docs/notes/megakernel_cuda_layer_sources.md` for MPK and Event Tensor reading
- `docs/todo/README.md` for open gaps
- `docs/in_progress/README.md` for active work
- `docs/design/README.md` for the implemented design index
