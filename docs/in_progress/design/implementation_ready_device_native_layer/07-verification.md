# Verification

Verification must prove runtime-linked behavior, not static metadata
construction.

## Build Checks

Required:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"
git diff --check
```

The build should prove:

- no materializer executable is built;
- no generated CUDA/C++ source is emitted;
- component runtime libraries link into orchestrate targets;
- example targets own their local CMake files.

## Unit Tests

Replace materialization tests with runtime component tests:

- dispatcher maps concrete `gemm_ar_problem` and `team_view` into tile counts
  and peer ranks;
- phased scheduler permits tile-ready communication without needing a
  co-resident progress guard;
- overlap scheduler rejects blocking communication without a valid progress
  guard;
- CUDA platform rejects device mismatch and invalid launch envelopes;
- NVSHMEM backend rejects invalid team and symmetric session mismatch;
- target runtime rejects unsupported team size, dtype, layout, and problem
  shape.

## Runtime Tests

Required local tests:

- direct ABI validation smoke with null stream;
- CUDA single-card numeric correctness;
- CUDA multi-card-capability smoke where hardware is available;
- two-rank NVSHMEM correctness with `nvshmrun` or Docker.

Required modes:

```text
golden local result
baseline phased single-card
baseline phased two-card
baseline overlap single-card
baseline overlap two-card
Megacu phased single-card
Megacu phased two-card
Megacu overlap single-card
Megacu overlap two-card
```

## Distributed Tests

Required before distributed support is claimed:

- `nvshmrun -np 2` single-host two-card run;
- MPI-launched adapter smoke or documented blocker;
- torch-distributed adapter smoke or documented blocker;
- negative tests for wrong PE count, device mismatch, and session mismatch.

## Inspection Checks

Search-based checks should fail the build or be run manually until automated:

```sh
rg -n "materialize_program|program_ir|owned_program_ir|dispatch_section|schedule_section|kernel_section|target_metadata" include src tests examples
```

After the runtime-linked replacement, these terms should not be implementation
dependencies. Historical design notes may still mention them only as rejected
architecture.
