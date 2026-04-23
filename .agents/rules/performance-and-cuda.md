# Performance And Platform Rules

- "Zero overhead" means no abstraction overhead beyond platform-native
  operations an expert would intentionally write by hand. Synchronization,
  atomics, polling, queues, and remote signals are algorithmic costs and must
  be named explicitly.
- Megacu shared layer must not be tightly bound to CUDA. Keep task, event,
  schedule, memory, and backend capability concepts platform-neutral unless a
  file lives under an explicit platform adapter.
- CUDA is the first platform and performance baseline. CUDA-specific code may
  use CUDA-native mechanisms and data structures directly, but those details
  belong under CUDA platform adapters, CUDA-specific backend adapters, examples,
  benchmarks, or tests.
- Static schedules are the default design direction. Dynamic scheduling is
  opt-in and must justify queue, atomic, and scheduler occupancy costs.
- Keep task and event representations compact enough to inspect in generated
  platform code, PTX/SASS, LLVM IR, or profiler traces as appropriate for the
  platform.
- Benchmark claims must state hardware, platform/runtime version, compiler
  flags, workload shape, baseline, and command.
- Preserve handwritten platform-native baselines for performance-sensitive
  abstractions. For the first backend, preserve handwritten CUDA/NVSHMEM
  baselines.
- If GPU hardware, CUDA, NVSHMEM, or profiler tooling is unavailable for the
  first backend, skip explicitly with the residual risk instead of inferring
  performance.
