# Program, Compile, Execute Flow

This document gives the concrete flow for using Megacu to construct a
mega-kernel from small operator kernels on CUDA+NVSHMEM.

## Program

A target author writes:

1. Plain operator kernels or operator functors. They receive raw pointers,
   scalars, descriptors, and a work cursor such as `tile_id`.
2. A recipe that submits operator tasks and sync-only tasks to an orch object.
3. A host-called orchestrate function that receives a driver and raw target
   args, prepares storage, and launches the linked runtime.

For GEMM-RS, the Megacu recipe is:

```text
for each GEMM output tile:
  submit GEMM producer task for that tile
  submit sync-only task that waits for that tile's team EventTensor
  submit reduce-scatter consumer task for that tile
```

For AG-GEMM, the recipe is:

```text
for each output N tile:
  create scheduler dependency group
  for each K tile needed by this N tile:
    submit all-gather producer task for that gathered B tile
    add returned producer task ref to the dependency group
  submit one sync-only task with:
    depends_on_many(dependency group)
    wait_strided(gathered EventTensor, N tile, K tile count, N tile stride)
  for each output M tile:
    submit GEMM consumer task for that output tile
```

The submitted tasks, not hidden subtasks inside a task, are the synchronization
level.

## Compile

The target links common Megacu components:

- runtime execution model: `host-orch` or `seeded-orch`;
- runtime loop: currently `block_tile`;
- scheduler: `explicit_asap`;
- dispatcher: `tile_grid` with `tile_grid` and `single_tile` attrs;
- EventTensor lowering: local CUDA or CUDA+NVSHMEM;
- platform/backend support: CUDA launch and NVSHMEM team/storage facts;
- target operator table.

There is no runtime build API. CMake/native build compiles the orchestrate
function, operators, and linked components into normal binaries or libraries.

## Execute

At execution time:

1. The launcher establishes process/rank facts. Direct, MPI, and Torch launch
   paths all normalize into the same CUDA+NVSHMEM driver shape.
2. The host orchestrate function allocates or receives raw target buffers and
   EventTensor storage.
3. `host-orch` builds a sealed arena before launch, or `seeded-orch` seeds the
   device-side construction path.
4. The host launches one CUDA mega-kernel.
5. Inside the mega-kernel, the runtime loop repeatedly asks the scheduler,
   dispatcher, and EventTensor component what can make progress.
6. Operator slots are invoked by the runtime. Operators do not signal/wait
   EventTensor state manually.
7. The host synchronizes the outer CUDA stream or device after the mega-kernel.

The CUDA stream is not an internal scheduler. It is only the host-side launch
and synchronization mechanism for the one mega-kernel.
