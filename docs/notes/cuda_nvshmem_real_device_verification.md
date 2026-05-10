# CUDA+NVSHMEM Real-Device Docker Verification

Date: 2026-05-09 Asia/Shanghai
Last updated: 2026-05-10 Asia/Shanghai

Purpose: verify the current PR branch on real CUDA devices through the shared
Docker CUDA+NVSHMEM environment, including direct, MPI, Torch, and two-rank
NVSHMEM validation paths. The verification must prove numerical correctness
against golden/baseline references, not only that programs return 0.

## Host Preconditions

Checked before the run:

```bash
git status --short --branch
docker --version
nvidia-smi -L
docker info --format '{{json .Runtimes}}'
free -h
df -h .
```

Observed host:

- Branch: `implementation/general-runtime-linked-components`
- Docker: `Docker version 27.2.0, build 3ab4256`
- GPUs visible to host:
  - GPU 0: NVIDIA A100 80GB PCIe
  - GPU 1: NVIDIA A100 80GB PCIe
  - GPU 2: NVIDIA A100 80GB PCIe
  - GPU 3: NVIDIA A100 80GB PCIe
  - GPU 4: NVIDIA A100-SXM4-80GB
  - GPU 5: NVIDIA A100 80GB PCIe
  - GPU 6: NVIDIA A100 80GB PCIe
- Free disk in repo filesystem: about 35T.
- Available memory: about 1.2T.

## Reproduce

From the repository root:

```bash
export MEGACU_DOCKER_GPUS='"device=0,1"'
export MEGACU_NVSHMEM_IMAGE='megacu-nvshmem:cuda12.8'
export MEGACU_NVSHMEM_BUILD_DIR='build-nvshmem'
tools/cuda_nvshmem/run_two_card_docker.sh
```

The script performs:

```bash
docker build -f docker/cuda_nvshmem/Dockerfile -t "${MEGACU_NVSHMEM_IMAGE}" .
docker run --rm --gpus "${MEGACU_DOCKER_GPUS}" --ipc=host \
  -v "${PWD}:/workspace/megacu" -w /workspace/megacu \
  "${MEGACU_NVSHMEM_IMAGE}" bash -lc "
    rm -rf '${MEGACU_NVSHMEM_BUILD_DIR}' &&
    cmake -S . -B '${MEGACU_NVSHMEM_BUILD_DIR}' -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DMEGACU_ENABLE_NVSHMEM_TESTS=ON &&
    cmake --build '${MEGACU_NVSHMEM_BUILD_DIR}' &&
    MEGACU_BUILD_DIR='${MEGACU_NVSHMEM_BUILD_DIR}' tools/cuda_nvshmem/run_direct.sh &&
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
      MEGACU_BUILD_DIR='${MEGACU_NVSHMEM_BUILD_DIR}' MEGACU_MPI_NP=2 \
      tools/cuda_nvshmem/run_mpi.sh &&
    MEGACU_BUILD_DIR='${MEGACU_NVSHMEM_BUILD_DIR}' MEGACU_SKIP_BUILD=1 \
      torchrun --standalone --nproc_per_node=2 tools/cuda_nvshmem/run_torch.py &&
    ctest --test-dir '${MEGACU_NVSHMEM_BUILD_DIR}' \
      -R 'cuda_gemm_allreduce_correctness|cuda_gemm_reduce_scatter_correctness|cuda_allgather_gemm_correctness|tiny_decode_correctness|nvshmem_two_rank_gemm_ar_correctness_(golden|manual_baseline|megacu)|nvshmem_two_rank_gemm_rs_correctness_megacu|nvshmem_two_rank_ag_gemm_correctness_megacu|mpi_two_rank_tiny_decode_correctness_megacu' \
      --output-on-failure"
```

## Result

Status: passed after tightening two-rank GEMM-RS and AG-GEMM checks to compare
Megacu outputs against golden and baseline reference outputs.

Evidence command run:

```bash
set -o pipefail
MEGACU_DOCKER_GPUS='"device=0,1"' \
MEGACU_NVSHMEM_IMAGE='megacu-nvshmem:cuda12.8' \
MEGACU_NVSHMEM_BUILD_DIR='build-nvshmem' \
tools/cuda_nvshmem/run_two_card_docker.sh 2>&1 | tee /tmp/megacu-cuda-nvshmem-docker-gate.log
```

The command exited with status 0. It was rerun after strengthening the
distributed GEMM-RS and AG-GEMM smokes.

Evidence log: `/tmp/megacu-cuda-nvshmem-docker-gate.log`

Focused regression command for the newly strict distributed golden/baseline
comparisons:

```bash
set -o pipefail
docker run --rm \
  --gpus '"device=0,1"' \
  --ipc=host \
  -v "${PWD}:/workspace/megacu" \
  -w /workspace/megacu \
  megacu-nvshmem:cuda12.8 \
  bash -lc "cmake --build build-nvshmem --target \
      megacu_mpi_two_rank_gemm_rs_smoke \
      megacu_mpi_two_rank_ag_gemm_smoke \
      megacu_nvshmem_two_rank_gemm_rs_smoke \
      megacu_nvshmem_two_rank_ag_gemm_smoke &&
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
      ctest --test-dir build-nvshmem \
        -R 'mpi_two_rank_(gemm_rs|ag_gemm)_correctness_megacu|nvshmem_two_rank_(gemm_rs|ag_gemm)_correctness_megacu' \
        --output-on-failure" \
  2>&1 | tee /tmp/megacu-distributed-reference-compare.log
```

Focused regression result: 8/8 CTest cases passed. Evidence log:
`/tmp/megacu-distributed-reference-compare.log`.

Observed Docker/CMake configuration:

- Docker image: `megacu-nvshmem:cuda12.8`
- Base image: `nvidia/cuda:12.8.0-devel-ubuntu22.04`
- CUDA compiler in container: NVIDIA `12.8.61`
- Build dir: `build-nvshmem`
- CMake generator: Ninja
- CMake flag: `-DMEGACU_ENABLE_NVSHMEM_TESTS=ON`
- Build type: `Debug`, so assert-backed value comparisons are active.
- GPUs passed to container: `device=0,1`

Observed build:

- Full `cmake --build build-nvshmem` completed.
- CUDA+NVSHMEM targets built, including:
  - GEMM-AllReduce golden, manual baseline, and Megacu paths;
  - GEMM-RS Megacu direct/MPI/Torch/NVSHMEM smoke binaries;
  - AG-GEMM Megacu direct/MPI/Torch/NVSHMEM smoke binaries;
  - tiny decode correctness and MPI smoke binaries.

Observed script phases:

- `tools/cuda_nvshmem/run_direct.sh`: passed 5 CTest cases:
  - `cuda_gemm_reduce_scatter_correctness`
  - `cuda_allgather_gemm_correctness`
  - `tiny_decode_correctness`
  - `mpi_two_rank_tiny_decode_correctness_megacu_host_orch`
  - `mpi_two_rank_tiny_decode_correctness_megacu_seeded_orch`
- `tools/cuda_nvshmem/run_mpi.sh`: passed 4 CTest cases:
  - `mpi_two_rank_gemm_rs_correctness_megacu_host_orch`
  - `mpi_two_rank_gemm_rs_correctness_megacu_seeded_orch`
  - `mpi_two_rank_ag_gemm_correctness_megacu_host_orch`
  - `mpi_two_rank_ag_gemm_correctness_megacu_seeded_orch`
- `torchrun --standalone --nproc_per_node=2 tools/cuda_nvshmem/run_torch.py`:
  returned successfully as part of the shell chain.
- Final CTest filter: passed 14 CTest cases:
  - `cuda_gemm_allreduce_correctness`
  - `nvshmem_two_rank_gemm_ar_correctness_golden`
  - `nvshmem_two_rank_gemm_ar_correctness_manual_baseline`
  - `nvshmem_two_rank_gemm_ar_correctness_megacu_host_orch`
  - `nvshmem_two_rank_gemm_ar_correctness_megacu_seeded_orch`
  - `cuda_gemm_reduce_scatter_correctness`
  - `nvshmem_two_rank_gemm_rs_correctness_megacu_host_orch`
  - `nvshmem_two_rank_gemm_rs_correctness_megacu_seeded_orch`
  - `cuda_allgather_gemm_correctness`
  - `nvshmem_two_rank_ag_gemm_correctness_megacu_host_orch`
  - `nvshmem_two_rank_ag_gemm_correctness_megacu_seeded_orch`
  - `tiny_decode_correctness`
  - `mpi_two_rank_tiny_decode_correctness_megacu_host_orch`
  - `mpi_two_rank_tiny_decode_correctness_megacu_seeded_orch`

Warning scan:

```bash
rg -n "warning|Warning|WARNING|UserWarning|deprecated|SM Arch|NumPy|OMP_NUM_THREADS|root user|no such option|failed to initialize|volatile destination|constexpr __host__" \
  /tmp/megacu-cuda-nvshmem-docker-gate.log
```

The scan returned no matches on the final run.

## Correctness Contract Audit

The Docker gate result is not interpreted as "exit code 0 only." The relevant
test sources copy outputs back and compare values as follows:

| Path | Real-device test | Correctness check |
| --- | --- | --- |
| GEMM-AllReduce, single GPU | `tests/runtime/cuda_gemm_allreduce_correctness.cc` | Runs golden, manual baseline, Megacu host-orch, and Megacu seeded-orch. Each output is copied back and compared against the same GEMM/allreduce expected values with `1.0e-4` tolerance. |
| GEMM-AllReduce, two rank | `tests/runtime/nvshmem_two_rank_smoke.cc` | Runs golden, manual baseline, Megacu host-orch, and Megacu seeded-orch modes through separate CTests. Each mode copies output back and checks every value equals the expected two-rank allreduce result. |
| GEMM-RS, single GPU | `tests/runtime/cuda_gemm_reduce_scatter_correctness.cc` | Runs golden, baseline, Megacu host-orch, and Megacu seeded-orch. Each output is copied back and compared against the same GEMM expected values with `1.0e-4` tolerance. |
| GEMM-RS, two rank | `tests/runtime/nvshmem_two_rank_gemm_rs_smoke.cc` | Runs golden and baseline implementations on an equivalent reduced reference input, then runs Megacu host-orch and Megacu seeded-orch. Each Megacu output is copied back and compared against both golden and baseline outputs with `1.0e-4` tolerance. |
| AG-GEMM, single GPU | `tests/runtime/cuda_allgather_gemm_correctness.cc` | Runs golden, baseline, Megacu host-orch, and Megacu seeded-orch. Each output is copied back and compared against the same GEMM expected values with `1.0e-4` tolerance. |
| AG-GEMM, two rank | `tests/runtime/nvshmem_two_rank_ag_gemm_smoke.cc` | Constructs a full gathered-B reference input, runs golden and baseline implementations, then runs Megacu host-orch and Megacu seeded-orch. Each Megacu output is copied back and compared against both golden and baseline outputs with `1.0e-4` tolerance. |
| Tiny decode, single GPU/local | `tests/runtime/tiny_decode_correctness.cc` | Runs golden, baseline, Megacu host-orch, Megacu seeded-orch, and compatibility Megacu path. It compares full buffers with `nearly_equal`. |
| Tiny decode, two rank MPI | `tests/runtime/mpi_two_rank_tiny_decode_smoke.cc` | Runs golden and the selected Megacu mode per rank, then compares full buffers with `nearly_equal`. |

Conclusion:

- The passing Docker gate proves numeric correctness for all covered paths; the
  tests do not merely check return values.
- Literal golden/baseline comparison coverage exists for single-GPU
  GEMM-AllReduce, GEMM-RS, AG-GEMM, tiny decode, two-rank GEMM-AllReduce,
  two-rank GEMM-RS, two-rank AG-GEMM, and two-rank tiny decode.
- The two-rank GEMM-RS smoke constructs golden and baseline reference outputs
  by running the existing golden/baseline implementations on an equivalent
  reduced reference input, then compares each Megacu mode against both outputs.
- The two-rank AG-GEMM smoke constructs a full gathered-B reference input, runs
  the existing golden/baseline implementations, then compares each Megacu mode
  against both outputs.
