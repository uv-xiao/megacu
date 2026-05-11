# Feature Task: Example Ergonomics

- Branch: `implementation/example-ergonomics`
- PR: draft
- Owner: Megacu contributors
- Status: design seed

## Goal

Make the CUDA+NVSHMEM examples substantially easier to read, write, and extend.
The current examples prove the runtime-linked architecture, but each example
still owns large `_arena.cuh`, `megacu.cu`, and `orchestrate.cc` files. That
shape makes Megacu look harder to use than it should be.

## Input

- Existing examples:
  - `examples/cuda_nvshmem/gemm_allreduce/`
  - `examples/cuda_nvshmem/gemm_reduce_scatter/`
  - `examples/cuda_nvshmem/allgather_gemm/`
  - `examples/cuda_nvshmem/tiny_decode_pipeline/`
- Current runtime design under `docs/design/`.
- User requirement: examples should be better, smaller, and should not make
  kernel programming feel hard because of repeated Megacu boilerplate.

## Output

- A common Megacu-owned example authoring surface for:
  - runtime arena storage;
  - host-orch and seeded-orch launch wrappers;
  - CUDA runtime operator table setup;
  - common driver/problem validation shape where appropriate.
- Examples that keep only workload-specific pieces:
  - target ABI;
  - recipe construction;
  - operator kernels;
  - golden/baseline references;
  - focused tutorial text.
- Tests that prove examples still use the operator-task + runtime-loop model
  and no longer duplicate heavy runtime scaffolding.

## Scope Checklist

- [x] Define input, output, and verification criteria
- [x] Write design seed before implementation
- [ ] Implement in coherent commits
- [ ] Verify locally
- [ ] Run Docker CUDA+NVSHMEM validation for distributed examples
- [ ] Sync `docs/design/`, `docs/todo/`, and `docs/in_progress/`

## Verification

Required before merge:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
MEGACU_DOCKER_GPUS='"device=0,1"' \
MEGACU_NVSHMEM_IMAGE='megacu-nvshmem:cuda12.8' \
MEGACU_NVSHMEM_BUILD_DIR='build-nvshmem' \
tools/cuda_nvshmem/run_two_card_docker.sh
```

Static checks should also prove the example-specific Megacu implementation
files shrink or disappear rather than being copied again.

## Tests

- Preserve numeric golden/baseline comparisons for all four examples.
- Add build/static contracts that reject example-owned generic runtime
  scaffolding when the helper API should own it.
- Keep both host-orch and seeded-orch paths covered.

## Docs

- Update each example `README.md` and `TUTORIAL.md` so a new contributor sees
  the small authoring surface first.
- Promote accepted API/design content into `docs/design/` when closing the PR.

## Closeout

- Remove stale in-progress task/design docs after accepted design content is
  promoted.
- Keep any future larger usability work in `docs/todo/`.
