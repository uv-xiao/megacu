#!/usr/bin/env python3
import os
import subprocess
import sys

import torch
import torch.distributed as dist


def broadcast_string(value, src):
    encoded = value.encode("ascii") if value is not None else b""
    length = torch.tensor([len(encoded)], dtype=torch.int64)
    dist.broadcast(length, src=src)

    payload_len = int(length.item())
    if dist.get_rank() == src:
        payload = torch.tensor(list(encoded), dtype=torch.uint8)
    else:
        payload = torch.empty(payload_len, dtype=torch.uint8)
    dist.broadcast(payload, src=src)
    return bytes(payload.tolist()).decode("ascii")


def run_uid_bootstrap_case(binary, mode, env, rank):
    if rank == 0:
        root = subprocess.Popen(
            [binary, "--uid-bootstrap-root", mode],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        uid = root.stdout.readline().strip()
        if not uid:
            root.kill()
            root.wait()
            raise RuntimeError("rank 0 did not publish an NVSHMEM UID")
    else:
        root = None
        uid = None

    uid = broadcast_string(uid, src=0)
    env = env.copy()
    env["MEGACU_NVSHMEM_UNIQUEID_HEX"] = uid
    dist.barrier()

    if rank == 0:
        root.stdin.write("\n")
        root.stdin.flush()
        return_code = root.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, [binary, "--uid-bootstrap-root", mode]
            )
    else:
        subprocess.check_call([binary, mode], env=env)


build_dir = os.environ.get("MEGACU_BUILD_DIR", "build")
adapter_binary = (
    f"{build_dir}/examples/cuda_nvshmem/gemm_allreduce/megacu_adapter_contracts"
)
gemm_rs_binary = (
    f"{build_dir}/examples/cuda_nvshmem/gemm_reduce_scatter/"
    "megacu_torch_uid_two_rank_gemm_rs_smoke"
)
ag_gemm_binary = (
    f"{build_dir}/examples/cuda_nvshmem/allgather_gemm/"
    "megacu_torch_uid_two_rank_ag_gemm_smoke"
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
                "--target",
                "megacu_torch_uid_two_rank_ag_gemm_smoke",
            ]
        )
    dist.barrier()

    if not os.path.exists(adapter_binary):
        raise SystemExit(f"missing adapter contract binary: {adapter_binary}")
    if not os.path.exists(gemm_rs_binary):
        raise SystemExit(f"missing Torch UID GEMM-RS binary: {gemm_rs_binary}")
    if not os.path.exists(ag_gemm_binary):
        raise SystemExit(f"missing Torch UID AG-GEMM binary: {ag_gemm_binary}")

    env = os.environ.copy()
    env["MEGACU_ADAPTER_CONTRACT_MODE"] = "torch"
    subprocess.check_call([adapter_binary], env=env)
    dist.barrier()

    for binary in (gemm_rs_binary, ag_gemm_binary):
        for mode in ("megacu_host_orch", "megacu_seeded_orch"):
            run_uid_bootstrap_case(binary, mode, env, rank)
            dist.barrier()
    dist.barrier()
finally:
    dist.destroy_process_group()
