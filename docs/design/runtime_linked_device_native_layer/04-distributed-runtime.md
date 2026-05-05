# Single-Host Runtime Resources

PR #4 needs 1-host/1-device and 1-host/2-device coverage. It does not need MPI
or torch-distributed adapters.

The architecture must still avoid baking CUDA/NVSHMEM handles into target
arguments. Platform and backend resources live in the driver; the linked
components read the resources they own.

## PR #4 Driver Model

All PR #4 launch paths produce:

```cpp
megacu::cuda_nvshmem::driver driver;
```

The driver contains component-owned resources:

```text
CUDA platform resources:
  stream
  device ordinal
  launch constraints needed by the phased target

NVSHMEM backend resources:
  team identity
  PE identity
  symmetric storage/session identity when team size is 2

Megacu target resources:
  none; target arguments, scratch/events, linked components, and capability
  envelopes belong to the host call or linked target, not to the driver
```

The driver includes only execution resources that must be supplied by the host
environment:

- CUDA stream and device/context identity, because host code uses them for
  asynchronous enqueue and synchronization.
- NVSHMEM PE/team identity and team size, because the backend needs them to map
  local and peer work.
- Distributed process facts such as local rank and launch mode, because future
  adapters need to normalize MPI, `nvshmrun`, or framework launch state without
  changing target arguments.

The driver excludes target arguments, target argument storage, operator
argument packs, target scratch/events, linked component sets, and capability
envelopes. Those are target-owned or call-owned facts. Keeping them out of the
driver prevents the driver from becoming an untyped runtime environment.

The workload entry is a host-called function that receives the driver plus
target arguments:

```cpp
extern "C" megacu::status gemm_allreduce_phased_orchestrate(
    megacu::cuda_nvshmem::driver &driver,
    const float *a,
    const float *b,
    float *partial,
    float *out,
    int m,
    int n,
    int k);
```

The caller or test harness builds the driver from CUDA/NVSHMEM resources before
invoking the target. The exact construction API is implementation detail for
PR #4. The key design rule is that `cuda::launch_view` and
`nvshmem::team_view` do not appear as target arguments, and target tensors and
scalars are not wrapped in a generic runtime bag.

## 1-Host/1-Device

For one GPU:

```text
device_count = 1
backend team size = 1
backend rank = 0
no remote symmetric peer required
```

The same orchestrate entry runs. The dispatcher maps submitted work to local
tiles only. The backend exposes that no remote communication resource is
required. The CUDA stream in the driver is the asynchronous execution surface
used by host code to enqueue and synchronize the target megakernel. Inside that
megakernel, the common device-side running loop dispatches ready work and
invokes linked operator kernels.

## 1-Host/2-Device

For two GPUs on one host:

```text
one process per GPU or an equivalent local launch mode
backend team size = 2
backend rank = current PE
symmetric partial/output/event storage is required
device-side NVSHMEM performs communication
```

The same orchestrate entry and direct target signature run. The dispatcher maps
the same submitted tasks to local plus peer work using backend resources from
the driver. The ASAP scheduler consumes the explicit dependency path from
compute to sync-only readiness to the phased fused communication operator.
PR #4 assumes the caller provides valid NVSHMEM launch state and symmetric
storage; it does not add a heavy fallback validator.

## Out Of Scope For PR #4

The following are future general implementation work:

- MPI-launched adapter;
- torch-distributed adapter;
- multi-node claims;
- arbitrary PE counts;
- host-side collectives as Megacu communication.

Future adapters should still produce the same driver resources and call the
same direct orchestrate entry. They must not reimplement dispatcher or
scheduler decisions.

## Capability Envelope

The PR #4 CUDA+NVSHMEM configuration should state:

```text
platform: CUDA
backend: NVSHMEM
launch modes: direct local process, nvshmrun or equivalent two-rank local run
devices: 1-host/1-device and 1-host/2-device
communication: device-side NVSHMEM for 2-device path
dtypes: f32 first proof
layouts: row-major contiguous first proof
scheduler: ASAP explicit-dependency only
unsupported: MPI, torch-distributed, arbitrary PE counts,
             multi-node performance claims, host-side collectives as Megacu
             communication
```

## Verification

Required before claiming PR #4 runtime coverage:

- 1-host/1-device numeric correctness for the phased Megacu path;
- 1-host/2-device numeric correctness for the phased Megacu path using
  device-side NVSHMEM.
