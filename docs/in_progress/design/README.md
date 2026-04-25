# In-Progress Design Docs

This directory holds active design work that is not ready for long-term
architecture docs yet.

Use a short top-level entry point for each workstream, then keep the detailed
chapters in one ordered subdirectory when the topic has enough depth to need
multiple files.

## Active Workstreams

- `concrete_impl/`: PR #3 implementation documentation for current
  `include/`, `src/`, CMake, and CUDA+NVSHMEM GEMM+AllReduce example call
  paths. This is implementation-facing documentation and must not be promoted
  into `docs/design/` until the PR is ready to merge.
