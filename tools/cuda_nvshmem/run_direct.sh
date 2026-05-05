#!/usr/bin/env bash
set -euo pipefail

build_dir="${MEGACU_BUILD_DIR:-build}"

cmake --build "${build_dir}" --target cuda_nvshmem_tiny_decode_correctness
ctest --test-dir "${build_dir}" -R 'tiny_decode_correctness' --output-on-failure
