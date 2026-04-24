cmake_minimum_required(VERSION 3.20)

project(megacu LANGUAGES CXX)

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
