# Overlap Baseline

This directory is the ownership point for the overlap pure CUDA/NVSHMEM
baseline. The current Megacu-free baseline entrypoints are implemented in
`../../golden/golden_gemm_allreduce.cu`:

- `golden_overlap_single_card_gemm_allreduce_f32`
- `golden_overlap_multi_card_gemm_allreduce_f32`

The baseline exists to check co-resident persistent semantics independently
from Megacu dispatch, scheduling, and lowering.
