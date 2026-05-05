#!/usr/bin/env python3
import os
import subprocess
import sys

import torch
import torch.distributed as dist


build_dir = os.environ.get("MEGACU_BUILD_DIR", "build")
required_env = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
missing = [name for name in required_env if not os.environ.get(name)]
if missing:
    raise SystemExit(
        "run_torch.py must be launched with torchrun; missing "
        + ", ".join(missing)
    )

if not dist.is_available():
    raise SystemExit("torch.distributed is required for the Torch adapter smoke")

dist.init_process_group(backend=os.environ.get("MEGACU_TORCH_BACKEND", "gloo"))
try:
    rank = dist.get_rank()
    if rank == 0 and os.environ.get("MEGACU_SKIP_BUILD") != "1":
        subprocess.check_call(
            ["cmake", "--build", build_dir, "--target", "megacu_adapter_contracts"]
        )
    dist.barrier()

    binary = f"{build_dir}/examples/cuda_nvshmem/gemm_allreduce/megacu_adapter_contracts"
    if not os.path.exists(binary):
        raise SystemExit(f"missing adapter contract binary: {binary}")

    env = os.environ.copy()
    env["MEGACU_ADAPTER_CONTRACT_MODE"] = "torch"
    subprocess.check_call([binary], env=env)
    dist.barrier()
finally:
    dist.destroy_process_group()
