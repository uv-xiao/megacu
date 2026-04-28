# Verification

PR #4 verification must prove the documentation split and the runtime-linked
architecture correction. It must not overclaim the future general concrete
implementation.

## Documentation Checks

Required for this split:

```sh
rg -n "docs/in_progress/design/concrete_impl|PR #4 must replace|concrete implementation docs for PR #4" \
  docs/in_progress docs/todo
git diff --check
```

The first check should find only historical human-word entries or deliberate
split-scope notes. Active architecture docs should point future concrete
implementation work to `docs/todo/concrete_impl/`.

## Architecture Review Checklist

Before PR #4 is considered ready for design review, check:

- no accepted path requires `program_ir`, `owned_program_ir`,
  `materialize_program`, static `dispatch_section`, static `schedule_section`,
  static `kernel_section`, or generated target metadata;
- direct ABI and typed runtime views are the visible entry point;
- dispatcher, scheduler, platform, backend, and target runtime ownership is
  clear at the architecture level;
- the tiny proof example is explicitly allowed to be problem-specific;
- problem-specific shortcuts are not promoted into public shared APIs;
- future generality requirements live in `docs/todo/concrete_impl/`;
- distributed launch and framework adapters are described as architecture
  boundaries without becoming hidden PR #4 implementation gates.

## Tiny Example Evidence

If PR #4 includes a runnable tiny example, collect focused evidence for:

- invalid runtime views return `megacu::status` before native launch;
- the linked native operator symbol is called without a materializer;
- runtime problem/team values drive mapping or work selection;
- no generated metadata section is needed;
- any problem-specific mapping or scheduling helper is example-local.

Exact commands depend on the final proof files. Record them in
`docs/in_progress/runtime_linked_megacu_slice.md` or the PR body when the proof
exists.

## Future General Implementation Evidence

The following checks belong to `docs/todo/concrete_impl/` and should not block
PR #4 unless the scope is explicitly expanded:

- reusable annotation-driven dispatcher tests;
- phased and overlap scheduler tests;
- CUDA platform validation tests;
- NVSHMEM backend validation tests;
- complete GEMM+AllReduce golden, baseline, and Megacu runtime tests;
- `nvshmrun`, MPI, and torch-distributed smoke tests;
- broad CMake build and CTest coverage for the full implementation.
