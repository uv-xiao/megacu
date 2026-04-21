# CUDA Performance Reviewer Profile

Use this profile when reviewing CUDA hot paths, generated CUDA, benchmarks, or
zero-overhead claims.

Check:

- the hot path uses CUDA-native mechanisms, not generic runtime indirection
- synchronization, atomics, polling, and queues are explicit and justified
- static scheduling is used unless dynamic scheduling has a measured reason
- task descriptors are compact and cache/shared-memory behavior is considered
- benchmark claims include hardware, compiler flags, workload, and baseline
- handwritten CUDA baselines exist for performance-sensitive abstractions

