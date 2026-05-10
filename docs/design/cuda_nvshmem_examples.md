# CUDA+NVSHMEM Examples

The implemented CUDA+NVSHMEM examples prove the same operator-task plus runtime
model across single-device, two-device, and local non-distributed scenarios.

## GEMM-RS

Path: `examples/cuda_nvshmem/gemm_reduce_scatter/`

The Megacu path uses GEMM tile producer tasks, per-tile sync-only EventTensor
tasks, and reduce-scatter tile consumer tasks. The same recipe is used for
`host-orch` and `seeded-orch`.

Validated paths:

- single host, one GPU correctness against golden and baseline;
- single host, two GPU NVSHMEM correctness;
- MPI-launched two-rank smoke;
- Torch `torchrun` UID-bootstrap two-rank smoke.

## AG-GEMM

Path: `examples/cuda_nvshmem/allgather_gemm/`

The Megacu path uses all-gather tile producer tasks, one sync-only task per
output N tile that waits for the arbitrary-K gathered B tile slice with
`depends_on_many` plus `wait_strided`, and GEMM consumer tasks for the output
M/N tiles. The same recipe is used for `host-orch` and `seeded-orch`.

Validated paths:

- single host, one GPU correctness against golden and baseline;
- single host, two GPU NVSHMEM correctness with uneven K ownership;
- MPI-launched two-rank smoke with uneven K ownership;
- Torch `torchrun` UID-bootstrap two-rank smoke with uneven K ownership.

## Tiny Decode Pipeline

Path: `examples/cuda_nvshmem/tiny_decode_pipeline/`

The tiny decode pipeline proves the same runtime execution model outside
distributed GEMM communication. It uses local CUDA EventTensor storage and
operator tasks for norm, projection, residual, MLP, and logits stages, compared
against golden and baseline implementations.

## Baselines

Golden and manual/native baselines remain separate from Megacu examples. They
exist for correctness and performance comparison, not as the Megacu
implementation path.
