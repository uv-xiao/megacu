# GEMM-AllReduce Megacu Runtime Variant

This variant composes GEMM tile operators and AllReduce tile operators into one
Megacu mega-kernel through the common runtime path.

## Flow

```text
host calls cuda_nvshmem_gemm_allreduce_{host_orch,seeded_orch}(driver, raw args)
  -> shared recipe records EventTensor + GEMM task + sync task + AllReduce task
  -> host-orch seals records on host, or seeded-orch publishes records on device
  -> runtime::loop::block_tile asks explicit_asap for ready tasks
  -> tile_grid gives each block tile work for operator tasks
  -> attr_event_tensor_i32 lowers notify/wait attrs for sync readiness
  -> operator table dispatches by op_slot
```

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target cuda_nvshmem_gemm_allreduce_megacu
```

## Test

```sh
ctest --test-dir build -R 'runtime_gemm_allreduce|cuda_gemm_allreduce_correctness' --output-on-failure
```

The direct correctness test compares golden, manual baseline, Megacu
host-orch, and Megacu seeded-orch on one host and one CUDA device.

## Distributed Run Envelope

The distributed 1-host-2-device path uses the same host-callable functions and
driver ABI. MPI and Torch launch adapters construct the driver; the recipe and
runtime path are unchanged. Docker is the source-of-truth environment for those
runs in this PR.
