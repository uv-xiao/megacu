# Example Organization Rules

- Examples live under `examples/<platform>_<backend>/<example>/`.
- Matching Docker assets live under `docker/<platform>_<backend>/<example>/`.
- Matching runnable helper scripts live under
  `tools/<platform>_<backend>/<example>/`.
- Use the same `<platform>_<backend>` and `<example>` names across
  `examples/`, `docker/`, and `tools/` so code, containers, and run scripts are
  discoverable as one unit.
- Every example owns its own `CMakeLists.txt`. The repository root may add the
  example with `add_subdirectory(...)`, but example libraries, golden targets,
  run targets, and example-specific tests must be declared by the example.
- Every example directory must include a `README.md` that explains:
  - what the example demonstrates;
  - the source file layout;
  - the execution path, with a simple visualization when useful;
  - simple pseudocode for the main algorithm or orchestration flow;
  - build, test, and run commands;
  - hardware, runtime, and backend assumptions;
  - known limitations or what the example intentionally does not prove.
- Every platform/backend grouping directory under `examples/`, `docker/`, and
  `tools/` must include a short `README.md` that lists contained examples and
  points to matching assets in the other roots.
- Do not add flat example names such as `examples/cuda_nvshmem_foo/` once a
  platform/backend grouping exists.
