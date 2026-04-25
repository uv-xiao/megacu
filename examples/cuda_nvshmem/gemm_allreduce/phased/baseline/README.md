# Phased Baseline

This directory is the ownership point for the phased pure CUDA/NVSHMEM
baseline. The current Megacu-free baseline entrypoints are implemented in
`../../golden/golden_gemm_allreduce.cu`:

- `golden_phased_single_card_gemm_allreduce_f32`
- `golden_phased_multi_card_gemm_allreduce_f32`

The baseline exists to check semantics independently from Megacu dispatch,
scheduling, and lowering.
