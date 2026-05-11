# TODO

`docs/todo/` tracks feature-sized gaps that are not implemented yet.

## Current Gaps

- [x] Agent harness scaffold
- [x] GitHub workflow skills for the agent harness
- [x] Megacu C++/CUDA layer design
- [x] Implementation-ready device-native design
- [x] Minimal build and verification tooling
- [x] General runtime-linked concrete implementation
- [ ] Example ergonomics and lightweight operator authoring surface
- [ ] Benchmark and profiling harness

Each future feature should have clear input, output, and verification criteria.

## Future Workstreams

- Benchmark and profiling harness with hardware, compiler, workload, and
  baseline evidence.
- Example ergonomics: remove repeated heavy `_arena.cuh`, `megacu.cu`, and
  `orchestrate.cc` boilerplate from each example by moving common runtime
  authoring machinery into Megacu-owned helpers while keeping kernels as thin
  operator tasks.
- Additional runtime loops and dispatcher strategies beyond the current
  block-tile CUDA examples.
- Larger framework packaging after the current Docker-backed Torch launch
  adapter remains stable.
