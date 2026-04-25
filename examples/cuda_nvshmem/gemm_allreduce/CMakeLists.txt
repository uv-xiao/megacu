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
  SOURCES gemm_allreduce_phased_orchestrate.cc
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE phased
  BACKEND_ENVELOPE TEAM_SIZE 2)

megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_overlap
  PROGRAM gemm_allreduce_overlap_program
  SOURCES gemm_allreduce_overlap_orchestrate.cc
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE co_resident_persistent
  BACKEND_ENVELOPE TEAM_SIZE 2)

if(MEGACU_ENABLE_NVSHMEM_TESTS)
  if(NOT MEGACU_HAS_CUDA_LANGUAGE)
    message(FATAL_ERROR
      "MEGACU_ENABLE_NVSHMEM_TESTS requires a CUDA compiler for numeric kernels")
  endif()

  find_package(CUDAToolkit REQUIRED)
  find_path(NVSHMEM_INCLUDE_DIR
    NAMES nvshmem.h
    HINTS ENV NVSHMEM_HOME
    PATHS /opt/nvidia/nvshmem /usr/include/nvshmem_12
    PATH_SUFFIXES include)
  find_library(NVSHMEM_HOST_LIBRARY
    NAMES libnvshmem_host.so.3 nvshmem_host
    HINTS ENV NVSHMEM_HOME
    PATHS /opt/nvidia/nvshmem /usr/lib/x86_64-linux-gnu/nvshmem/12
    PATH_SUFFIXES lib lib64)
  find_library(NVSHMEM_DEVICE_LIBRARY
    NAMES libnvshmem_device.a nvshmem_device
    HINTS ENV NVSHMEM_HOME
    PATHS /opt/nvidia/nvshmem /usr/lib/x86_64-linux-gnu/nvshmem/12
    PATH_SUFFIXES lib lib64)
  find_program(NVSHMEMRUN_EXECUTABLE
    NAMES nvshmrun nvshmemrun
    HINTS ENV NVSHMEM_HOME
    PATHS /usr/bin/nvshmem_12 /opt/nvidia/nvshmem
    PATH_SUFFIXES bin)

  if(NOT NVSHMEM_INCLUDE_DIR OR
     NOT NVSHMEM_HOST_LIBRARY OR
     NOT NVSHMEM_DEVICE_LIBRARY OR
     NOT NVSHMEMRUN_EXECUTABLE)
    message(FATAL_ERROR
      "MEGACU_ENABLE_NVSHMEM_TESTS requires nvshmem.h, libnvshmem_host, libnvshmem_device, and nvshmrun")
  endif()
endif()

if(MEGACU_HAS_CUDA_LANGUAGE)
  add_library(golden_cuda_nvshmem_gemm_allreduce STATIC
    golden/golden_gemm_allreduce.cu)
  target_link_libraries(golden_cuda_nvshmem_gemm_allreduce PUBLIC
    CUDA::cudart)
  target_include_directories(golden_cuda_nvshmem_gemm_allreduce PUBLIC
    ${PROJECT_SOURCE_DIR})
  if(CMAKE_VERSION VERSION_LESS 3.25)
    target_compile_options(golden_cuda_nvshmem_gemm_allreduce PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
    set(MEGACU_GEMM_AR_CUDA_STANDARD_PROPERTIES "")
  else()
    set(MEGACU_GEMM_AR_CUDA_STANDARD_PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON)
  endif()
  set_target_properties(golden_cuda_nvshmem_gemm_allreduce PROPERTIES
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
    ${MEGACU_GEMM_AR_CUDA_STANDARD_PROPERTIES})

  if(MEGACU_ENABLE_NVSHMEM_TESTS)
    target_compile_definitions(golden_cuda_nvshmem_gemm_allreduce PRIVATE
      MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM=1)
    target_include_directories(golden_cuda_nvshmem_gemm_allreduce PUBLIC
      ${NVSHMEM_INCLUDE_DIR})
    target_link_libraries(golden_cuda_nvshmem_gemm_allreduce PUBLIC
      CUDA::cuda_driver
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
  endif()

  add_library(megacu_gemm_allreduce_cuda_native OBJECT
    megacu/megacu_gemm_allreduce_kernels.cu)
  target_link_libraries(megacu_gemm_allreduce_cuda_native PUBLIC
    megacu_headers
    golden_cuda_nvshmem_gemm_allreduce
    CUDA::cudart)
  if(CMAKE_VERSION VERSION_LESS 3.25)
    target_compile_options(megacu_gemm_allreduce_cuda_native PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
  endif()
  set_target_properties(megacu_gemm_allreduce_cuda_native PROPERTIES
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
    ${MEGACU_GEMM_AR_CUDA_STANDARD_PROPERTIES})

  foreach(target
      cuda_nvshmem_gemm_allreduce_phased
      cuda_nvshmem_gemm_allreduce_overlap)
    target_sources(${target} PRIVATE
      $<TARGET_OBJECTS:megacu_gemm_allreduce_cuda_native>)
    target_link_libraries(${target} PUBLIC
      golden_cuda_nvshmem_gemm_allreduce
      CUDA::cudart)
    target_compile_definitions(${target} PRIVATE
      MEGACU_HAS_CUDA_NUMERIC_PATH=1)
  endforeach()
endif()

add_executable(megacu_compile_gemm_allreduce_descriptor
  ${PROJECT_SOURCE_DIR}/tests/build/compile_gemm_allreduce_descriptor.cc)
target_link_libraries(megacu_compile_gemm_allreduce_descriptor PRIVATE
  megacu_headers)

add_test(
  NAME compile_gemm_allreduce_descriptor
  COMMAND megacu_compile_gemm_allreduce_descriptor)

add_executable(megacu_materialize_gemm_allreduce
  ${PROJECT_SOURCE_DIR}/tests/build/materialize_gemm_allreduce.cc)
target_link_libraries(megacu_materialize_gemm_allreduce PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME materialize_gemm_allreduce
  COMMAND megacu_materialize_gemm_allreduce)

add_executable(megacu_component_metadata_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/component_metadata_contracts.cc)
target_link_libraries(megacu_component_metadata_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME component_metadata_contracts
  COMMAND megacu_component_metadata_contracts)

add_executable(megacu_runtime_adapter_validation
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_adapter_validation.cc)
target_link_libraries(megacu_runtime_adapter_validation PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_adapter_validation
  COMMAND megacu_runtime_adapter_validation)

add_executable(megacu_linked_metadata_symbols
  ${PROJECT_SOURCE_DIR}/tests/build/linked_metadata_symbols.cc)
target_link_libraries(megacu_linked_metadata_symbols PRIVATE
  megacu_headers
  cuda_nvshmem_gemm_allreduce_phased
  cuda_nvshmem_gemm_allreduce_overlap)

add_test(
  NAME linked_metadata_symbols
  COMMAND megacu_linked_metadata_symbols)

add_executable(megacu_direct_orchestrate_smoke
  ${PROJECT_SOURCE_DIR}/tests/runtime/direct_orchestrate_smoke.cc)
target_link_libraries(megacu_direct_orchestrate_smoke PRIVATE
  megacu_headers
  cuda_nvshmem_gemm_allreduce_phased
  cuda_nvshmem_gemm_allreduce_overlap)

add_test(
  NAME direct_orchestrate_smoke
  COMMAND megacu_direct_orchestrate_smoke)

if(CUDAToolkit_FOUND)
  add_executable(megacu_cuda_orchestrate_smoke
    ${PROJECT_SOURCE_DIR}/tests/runtime/cuda_orchestrate_smoke.cc)
  target_link_libraries(megacu_cuda_orchestrate_smoke PRIVATE
    megacu_headers
    cuda_nvshmem_gemm_allreduce_phased
    cuda_nvshmem_gemm_allreduce_overlap
    CUDA::cudart)

  add_test(
    NAME cuda_orchestrate_smoke
    COMMAND megacu_cuda_orchestrate_smoke)

  add_executable(megacu_cuda_multicard_smoke
    ${PROJECT_SOURCE_DIR}/tests/runtime/cuda_multicard_smoke.cc)
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

  if(MEGACU_HAS_CUDA_LANGUAGE)
    add_executable(megacu_cuda_gemm_allreduce_correctness
      ${PROJECT_SOURCE_DIR}/tests/runtime/cuda_gemm_allreduce_correctness.cc)
    target_link_libraries(megacu_cuda_gemm_allreduce_correctness PRIVATE
      megacu_headers
      golden_cuda_nvshmem_gemm_allreduce
      cuda_nvshmem_gemm_allreduce_phased
      cuda_nvshmem_gemm_allreduce_overlap
      CUDA::cudart)

    add_test(
      NAME cuda_gemm_allreduce_correctness
      COMMAND megacu_cuda_gemm_allreduce_correctness)
  endif()
endif()

if(MEGACU_ENABLE_NVSHMEM_TESTS)
  add_executable(megacu_nvshmem_two_rank_smoke
    ${PROJECT_SOURCE_DIR}/tests/runtime/nvshmem_two_rank_smoke.cc)
  target_link_libraries(megacu_nvshmem_two_rank_smoke PRIVATE
    megacu_headers
    golden_cuda_nvshmem_gemm_allreduce
    cuda_nvshmem_gemm_allreduce_phased
    cuda_nvshmem_gemm_allreduce_overlap
    CUDA::cudart
    CUDA::cuda_driver
    ${NVSHMEM_DEVICE_LIBRARY}
    ${NVSHMEM_HOST_LIBRARY})

  foreach(mode
      golden_phased
      golden_overlap
      megacu_phased
      megacu_overlap)
    add_test(
      NAME nvshmem_two_rank_smoke_${mode}
      COMMAND ${NVSHMEMRUN_EXECUTABLE} -np 2
              $<TARGET_FILE:megacu_nvshmem_two_rank_smoke> ${mode})
    set_tests_properties(nvshmem_two_rank_smoke_${mode} PROPERTIES
      SKIP_RETURN_CODE 77)
  endforeach()
endif()
