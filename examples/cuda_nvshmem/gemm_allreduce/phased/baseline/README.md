# Phased Baseline

This directory is the ownership point for the phased pure CUDA/NVSHMEM
baseline.

`manual_megakernel_gemm_allreduce.cu` contains the handwritten fused
mega-kernel baseline. It directly spells out:

- GEMM tile computation;
- event publication/acquire;
- AllReduce tile consumption;
- the final fused kernel body that sequences those steps.

That manual fused body intentionally lives here, not in `../megacu/`. The
Megacu example should compose operator tasks and event tensor handlers through
the Megacu CUDA+NVSHMEM lowering path.

The current Megacu-free golden entrypoints are implemented in
`../../golden/golden_gemm_allreduce.cu`:

- `golden_phased_single_card_gemm_allreduce_f32`
- `golden_phased_multi_card_gemm_allreduce_f32`

The baselines exist to check semantics independently from the Megacu
runtime-linked dispatcher, scheduler, event tensor, and driver path.
