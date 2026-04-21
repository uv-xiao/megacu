# Megacu

Megacu is an early-stage project for a thin C++/CUDA layer for building
megakernels close to native CUDA. The core performance goal is zero abstraction
overhead on the hot path: users should write normal CUDA/CuTe/CUTLASS/NVSHMEM
device code, while Megacu supplies explicit task, event, and scheduling
building blocks.

The first project invariant is process quality: agent-friendly workflow,
recorded source readings, explicit design, and verification evidence come
before implementation.

Start with:

- `AGENTS.md` for agent operating rules
- `docs/notes/megakernel_cuda_layer_sources.md` for MPK and Event Tensor reading
- `docs/todo/README.md` for open gaps
- `docs/in_progress/README.md` for active work
- `docs/design/README.md` for implemented design

