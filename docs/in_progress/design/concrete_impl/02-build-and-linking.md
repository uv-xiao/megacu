# Build And Linking

Megacu compilation in the current implementation means normal C++/CUDA compile
and link. It does not generate new CUDA, C++, PTX, or another intermediate
programming language.

## Component Library

`examples/cuda_nvshmem/gemm_allreduce/CMakeLists.txt` calls:

```cmake
megacu_add_components(
  NAME cuda_nvshmem_static
  DISPATCHER tiled_compute_comm_dispatch
  SCHEDULER static_persistent
  KERNEL_LOWERING persistent_stitch
  PLATFORM cuda
  BACKEND nvshmem)
```

`cmake/MegacuTargets.cmake` expands this into a static library containing:

```text
src/program/materialize.cc
src/dispatcher/tiled_compute_comm_dispatch.cc
src/scheduler/static_persistent.cc
src/lowering/persistent_stitch.cc
src/platform/cuda/platform.cc
src/platform/cuda/validation.cc
src/backends/nvshmem/backend.cc
src/backends/nvshmem/validation.cc
src/target/runtime.cc
```

The CMake arguments are currently recorded as compile definitions and target
properties:

```text
MEGACU_COMPONENT_DISPATCHER
MEGACU_COMPONENT_SCHEDULER
MEGACU_COMPONENT_KERNEL_LOWERING
MEGACU_COMPONENT_PLATFORM
MEGACU_COMPONENT_BACKEND
```

Those properties make the selected configuration inspectable. They do not yet
select different source files from a registry; `megacu_add_components` links the
same first-slice source files and validates that each configuration name was
provided.

## Orchestrate Targets

Each Megacu variant owns its own `CMakeLists.txt`:

```text
examples/cuda_nvshmem/gemm_allreduce/phased/megacu/CMakeLists.txt
examples/cuda_nvshmem/gemm_allreduce/overlap/megacu/CMakeLists.txt
```

Each file calls `megacu_add_orchestrate_target`, for example:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_overlap
  PROGRAM gemm_allreduce_overlap_program
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE co_resident_persistent
  SOURCES gemm_allreduce_overlap_orchestrate.cc)
```

The function creates a static library for the orchestrate target and links:

```text
megacu_headers
cuda_nvshmem_static
```

The example `CMakeLists.txt` then adds CUDA object files and native
CUDA/NVSHMEM support:

```text
megacu_gemm_allreduce_cuda_native
golden_cuda_nvshmem_gemm_allreduce
CUDA::cudart
optional CUDA::cuda_driver + NVSHMEM host/device libraries
```

## What Is Linked

For the phased target:

```text
cuda_nvshmem_gemm_allreduce_phased
  includes gemm_allreduce_phased_orchestrate.cc
  links cuda_nvshmem_static
  links megacu_gemm_allreduce_cuda_native
  links golden_cuda_nvshmem_gemm_allreduce
```

For the overlap target:

```text
cuda_nvshmem_gemm_allreduce_overlap
  includes gemm_allreduce_overlap_orchestrate.cc
  links cuda_nvshmem_static
  links megacu_gemm_allreduce_cuda_native
  links golden_cuda_nvshmem_gemm_allreduce
```

The linked native object currently contains:

```text
phased/megacu/megacu_gemm_allreduce_phased.cu
overlap/megacu/megacu_gemm_allreduce_overlap.cu
```

These files provide the C symbols called by the orchestrate common path:

```text
megacu_cuda_gemm_allreduce_phased_f32
megacu_cuda_gemm_allreduce_overlap_f32
```

## What Is Not Generated

The current Megacu build does not:

- emit generated CUDA files;
- emit generated C++ orchestrate code;
- translate `program_builder` records into new kernels;
- synthesize CMake targets from metadata at build time;
- create PTX/cubin/fatbin artifacts as checked-in outputs.

The build links existing implementation files to high-level APIs. This matches
the design requirement that kernel-lowering and target-lowering connect
existing low-level implementation symbols to Megacu metadata instead of
generating a new source language.

## Current Build Limitation

`megacu_add_orchestrate_target` records the program name and scheduler mode as
CMake metadata, but the checked-in orchestrate source still manually defines
the linked `metadata_header` constants. A stronger implementation should make
that header mechanically derived from `materialize_program<Program>(options)`
or otherwise prove that CMake target properties and linked metadata cannot
drift.
