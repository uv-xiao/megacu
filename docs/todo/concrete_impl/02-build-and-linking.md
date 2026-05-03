# Build And Linking

Megacu compilation means normal C++/CUDA compile and link. It does not mean
translating an authored program into IR, static sections, generated C++, or
generated CUDA.

## ConfigureTarget

The CMake API should move from `megacu_add_components` toward an explicit
`megacu_add_configure_target` shape:

```cmake
megacu_add_configure_target(
  NAME cuda_nvshmem_static
  DISPATCHER annotated_runtime
  SCHEDULER static_phased_or_overlap
  PLATFORM cuda
  BACKEND nvshmem)
```

This target links reusable runtime component implementations:

```text
src/dispatcher/annotated_runtime.cc
src/scheduler/static_runtime.cc
src/platform/cuda/validation.cc
src/backends/nvshmem/runtime.cc
src/backends/nvshmem/validation.cc
src/target/runtime.cc
```

It must not link:

```text
src/program/materialize.cc
src/lowering/persistent_stitch.cc
metadata section builders as the main execution contract
```

The dispatcher is part of the `ConfigureTarget`, but it is general. It is not
named `gemm_ar` and it does not contain example-specific policy.

## OrchTarget

Each Megacu example links one direct target. PR #4 keeps only the phased
GEMM+AllReduce target active:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_phased
  CONFIGURE_TARGET cuda_nvshmem_static
  SOURCES gemm_allreduce_phased_orchestrate.cc
  OPERATORS megacu_cuda_gemm_allreduce_phased_f32
  CAPABILITY cuda_nvshmem_gemm_allreduce_phased_capability)
```

The `OrchTarget` source supplies:

- the direct ABI;
- workload problem/workspace types;
- virtual participant annotations;
- native operator symbols;
- workload capability facts.

The `ConfigureTarget` supplies:

- dispatcher algorithm;
- scheduler implementation;
- platform validation;
- backend validation and primitives;
- common target runtime.

## What Is Linked

For the first CUDA+NVSHMEM configuration:

```text
cuda_nvshmem_static
  -> annotated runtime dispatcher
  -> explicit ASAP scheduler runtime
  -> CUDA platform validation
  -> NVSHMEM backend validation and primitive wrappers
  -> common target runtime

cuda_nvshmem_gemm_allreduce_phased
  -> cuda_nvshmem_static
  -> phased OrchTarget source
  -> phased Megacu native operator
  -> common GEMM+AllReduce problem/workspace helpers

```

Golden and baseline libraries are test/example dependencies. The Megacu
orchestrate targets should not call `golden/` functions as their production
operator path.

## Current CMake Gap

`cmake/MegacuTargets.cmake` currently links the materialization source and
metadata section builders through `megacu_add_components`. The future general
implementation update should:

1. add or rename to `megacu_add_configure_target`;
2. link runtime component source files instead of materializer files;
3. keep target properties for inspectability;
4. make unknown dispatcher/scheduler/platform/backend names fail at configure
   time;
5. make each example variant own its own `CMakeLists.txt`;
6. remove compile definitions that imply kernel lowering or static section
   generation.

## What Must Not Be Generated

The build must not:

- emit generated CUDA or C++ sources;
- generate per-program binary metadata;
- create a materializer executable as a normal path;
- translate C++ participant annotations into a new language;
- select backend or scheduler through runtime string loading.
