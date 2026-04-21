# Performance And CUDA Rules

- "Zero overhead" means no abstraction overhead beyond CUDA operations an expert
  would intentionally write by hand. Synchronization, atomics, polling, queues,
  and remote signals are algorithmic costs and must be named explicitly.
- Prefer CUDA-native mechanisms and data structures over generic runtimes in
  hot paths.
- Static schedules are the default design direction. Dynamic scheduling is
  opt-in and must justify queue, atomic, and scheduler occupancy costs.
- Keep task and event representations compact enough to inspect in generated
  CUDA, PTX/SASS, or profiler traces.
- Benchmark claims must state hardware, CUDA version, compiler flags, workload
  shape, baseline, and command.
- Preserve handwritten CUDA baselines for performance-sensitive abstractions.
- If GPU hardware, CUDA, NVSHMEM, or profiler tooling is unavailable, skip
  explicitly with the residual risk instead of inferring performance.

