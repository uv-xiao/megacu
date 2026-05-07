#!/usr/bin/env python3
import os
import subprocess
import sys

import torch
import torch.distributed as dist


build_dir = os.environ.get("MEGACU_BUILD_DIR", "build")
adapter_binary = (
    f"{build_dir}/examples/cuda_nvshmem/gemm_allreduce/megacu_adapter_contracts"
)
gemm_rs_binary = (
    f"{build_dir}/examples/cuda_nvshmem/gemm_reduce_scatter/"
    "megacu_torch_uid_two_rank_gemm_rs_smoke"
)
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
            [
                "cmake",
                "--build",
                build_dir,
                "--target",
                "megacu_adapter_contracts",
                "--target",
                "megacu_torch_uid_two_rank_gemm_rs_smoke",
            ]
        )
    dist.barrier()

    if not os.path.exists(adapter_binary):
        raise SystemExit(f"missing adapter contract binary: {adapter_binary}")
    if not os.path.exists(gemm_rs_binary):
        raise SystemExit(f"missing Torch UID GEMM-RS binary: {gemm_rs_binary}")

    env = os.environ.copy()
    env["MEGACU_ADAPTER_CONTRACT_MODE"] = "torch"
    subprocess.check_call([adapter_binary], env=env)
    dist.barrier()

    uid = [None]
    if rank == 0:
        uid[0] = subprocess.check_output(
            [gemm_rs_binary, "--print-uid"], text=True
        ).strip()
    dist.broadcast_object_list(uid, src=0)
    env["MEGACU_NVSHMEM_UNIQUEID_HEX"] = uid[0]

    for mode in ("megacu_host_orch", "megacu_seeded_orch"):
        subprocess.check_call([gemm_rs_binary, mode], env=env)
        dist.barrier()
    dist.barrier()
finally:
    dist.destroy_process_group()
