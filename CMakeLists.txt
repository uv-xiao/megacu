cmake_minimum_required(VERSION 3.20)

project(megacu LANGUAGES CXX)

option(MEGACU_ENABLE_NVSHMEM_TESTS
  "Build CUDA+NVSHMEM two-rank runtime validation tests" OFF)

find_package(CUDAToolkit QUIET)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MegacuTargets)

add_library(megacu_headers INTERFACE)
target_include_directories(megacu_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(megacu_headers INTERFACE cxx_std_20)

megacu_add_components(
  NAME cuda_nvshmem_static
  DISPATCHER tiled_compute_comm_dispatch
  SCHEDULER static_persistent
  KERNEL_LOWERING persistent_stitch
  PLATFORM cuda
  BACKEND nvshmem)

megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_phased
  PROGRAM gemm_allreduce_phased_program
  SOURCES examples/cuda_nvshmem_gemm_allreduce/gemm_allreduce_orchestrate.cc
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE phased
  BACKEND_ENVELOPE TEAM_SIZE 2)

megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_overlap
  PROGRAM gemm_allreduce_overlap_program
  SOURCES examples/cuda_nvshmem_gemm_allreduce/gemm_allreduce_orchestrate.cc
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE co_resident_persistent
  BACKEND_ENVELOPE TEAM_SIZE 2)

enable_testing()

add_executable(megacu_compile_gemm_allreduce_descriptor
  tests/build/compile_gemm_allreduce_descriptor.cc)
target_link_libraries(megacu_compile_gemm_allreduce_descriptor PRIVATE
  megacu_headers)

add_test(
  NAME compile_gemm_allreduce_descriptor
  COMMAND megacu_compile_gemm_allreduce_descriptor)

add_executable(megacu_materialize_gemm_allreduce
  tests/build/materialize_gemm_allreduce.cc)
target_link_libraries(megacu_materialize_gemm_allreduce PRIVATE
  megacu_headers)

add_test(
  NAME materialize_gemm_allreduce
  COMMAND megacu_materialize_gemm_allreduce)

add_executable(megacu_direct_orchestrate_smoke
  tests/runtime/direct_orchestrate_smoke.cc)
target_link_libraries(megacu_direct_orchestrate_smoke PRIVATE
  megacu_headers
  cuda_nvshmem_gemm_allreduce_phased
  cuda_nvshmem_gemm_allreduce_overlap)

add_test(
  NAME direct_orchestrate_smoke
  COMMAND megacu_direct_orchestrate_smoke)

if(CUDAToolkit_FOUND)
  add_executable(megacu_cuda_orchestrate_smoke
    tests/runtime/cuda_orchestrate_smoke.cc)
  target_link_libraries(megacu_cuda_orchestrate_smoke PRIVATE
    megacu_headers
    cuda_nvshmem_gemm_allreduce_phased
    cuda_nvshmem_gemm_allreduce_overlap
    CUDA::cudart)

  add_test(
    NAME cuda_orchestrate_smoke
    COMMAND megacu_cuda_orchestrate_smoke)

  add_executable(megacu_cuda_multicard_smoke
    tests/runtime/cuda_multicard_smoke.cc)
  target_link_libraries(megacu_cuda_multicard_smoke PRIVATE
    megacu_headers
    cuda_nvshmem_gemm_allreduce_phased
    cuda_nvshmem_gemm_allreduce_overlap
    CUDA::cudart)

  add_test(
    NAME cuda_multicard_smoke
    COMMAND megacu_cuda_multicard_smoke)
  set_tests_properties(cuda_multicard_smoke PROPERTIES
    SKIP_RETURN_CODE 77)
endif()

if(MEGACU_ENABLE_NVSHMEM_TESTS)
  find_package(CUDAToolkit REQUIRED)
  find_path(NVSHMEM_INCLUDE_DIR
    NAMES nvshmem.h
    HINTS ENV NVSHMEM_HOME
    PATHS /usr/include/nvshmem_12
    PATH_SUFFIXES include)
  find_library(NVSHMEM_HOST_LIBRARY
    NAMES nvshmem_host
    HINTS ENV NVSHMEM_HOME
    PATHS /usr/lib/x86_64-linux-gnu/nvshmem/12
    PATH_SUFFIXES lib lib64)
  find_program(NVSHMEMRUN_EXECUTABLE
    NAMES nvshmrun nvshmemrun
    HINTS ENV NVSHMEM_HOME
    PATHS /usr/bin/nvshmem_12
    PATH_SUFFIXES bin)

  if(NOT NVSHMEM_INCLUDE_DIR OR
     NOT NVSHMEM_HOST_LIBRARY OR
     NOT NVSHMEMRUN_EXECUTABLE)
    message(FATAL_ERROR
      "MEGACU_ENABLE_NVSHMEM_TESTS requires nvshmem.h, libnvshmem_host, and nvshmrun")
  endif()

  add_executable(megacu_nvshmem_two_rank_smoke
    tests/runtime/nvshmem_two_rank_smoke.cc)
  target_link_libraries(megacu_nvshmem_two_rank_smoke PRIVATE
    megacu_headers
    cuda_nvshmem_gemm_allreduce_phased
    cuda_nvshmem_gemm_allreduce_overlap
    CUDA::cudart
    CUDA::cuda_driver
    ${NVSHMEM_HOST_LIBRARY})

  add_test(
    NAME nvshmem_two_rank_smoke
    COMMAND ${NVSHMEMRUN_EXECUTABLE} -np 2
            $<TARGET_FILE:megacu_nvshmem_two_rank_smoke>)
  set_tests_properties(nvshmem_two_rank_smoke PROPERTIES
    SKIP_RETURN_CODE 77)
endif()
