#!/usr/bin/env bash
set -euo pipefail

build_dir="${MEGACU_BUILD_DIR:-build}"
mpi_np="${MEGACU_MPI_NP:-2}"
binary="${build_dir}/examples/cuda_nvshmem/gemm_allreduce/megacu_adapter_contracts"

if ! command -v mpirun >/dev/null 2>&1; then
  echo "mpirun is required for the MPI adapter smoke" >&2
  exit 1
fi

cmake --build "${build_dir}" --target megacu_adapter_contracts
if [[ ! -x "${binary}" ]]; then
  echo "missing adapter contract binary: ${binary}" >&2
  exit 1
fi

mpirun -np "${mpi_np}" \
  env MEGACU_ADAPTER_CONTRACT_MODE=mpi "${binary}"

if [[ "${MEGACU_RUN_EXAMPLE_TESTS:-1}" == "1" ]]; then
  cmake --build "${build_dir}" --target megacu_mpi_two_rank_gemm_rs_smoke
  ctest --test-dir "${build_dir}" \
    -R 'mpi_two_rank_gemm_rs_correctness_megacu' \
    --output-on-failure
fi
