# Human Words: Megakernel Design

## Category

- Primary: Megakernel Design

## Timeline

- 2026-04-22 Asia/Shanghai - Multi-GPU from the beginning
  > Multi-gpu from the beginning.
  - Context: User answered the first Megacu design-scope question about whether
    the first executable slice should target single-GPU static scheduling only
    or include multi-GPU/NVSHMEM concepts from the beginning.
  - Related: `docs/in_progress/megacu_cpp_cuda_layer.md`,
    `docs/in_progress/design/megacu_cpp_cuda_layer.md`
  - Agent interpretation: The initial architecture must model multi-GPU
    ownership, communication, and verification from day one rather than adding
    them as a later retrofit.
