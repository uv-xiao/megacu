# TODO

`docs/todo/` tracks feature-sized gaps that are not implemented yet.

## Current Gaps

- [x] Agent harness scaffold
- [x] GitHub workflow skills for the agent harness
- [x] Megacu C++/CUDA layer design
- [x] Implementation-ready device-native design
- [x] Minimal build and verification tooling
- [ ] First executable persistent-kernel slice
- [ ] General runtime-linked concrete implementation
- [ ] Benchmark and profiling harness

Each future feature should have clear input, output, and verification criteria.

## Future Workstreams

- `concrete_impl/`: future implementation documentation for a general
  runtime-linked Megacu implementation. This work was split out of PR #4 on
  2026-04-28 so PR #4 can focus on architecture repair and one tiny
  problem-specific proof example.
