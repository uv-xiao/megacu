if(MEGACU_ENABLE_NVSHMEM_TESTS)
  find_package(CUDAToolkit REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)
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
  set(MEGACU_GEMM_RS_CUDA_STANDARD_PROPERTIES "")
  add_library(golden_cuda_nvshmem_gemm_reduce_scatter STATIC
    golden/golden_gemm_reduce_scatter.cu)
  target_link_libraries(golden_cuda_nvshmem_gemm_reduce_scatter PUBLIC
    megacu_headers
    CUDA::cudart)
  target_include_directories(golden_cuda_nvshmem_gemm_reduce_scatter PUBLIC
    ${PROJECT_SOURCE_DIR})
  target_compile_options(golden_cuda_nvshmem_gemm_reduce_scatter PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  add_library(baseline_cuda_nvshmem_gemm_reduce_scatter STATIC
    baseline/baseline_gemm_reduce_scatter.cu)
  target_link_libraries(baseline_cuda_nvshmem_gemm_reduce_scatter PUBLIC
    megacu_headers
    CUDA::cudart)
  target_include_directories(baseline_cuda_nvshmem_gemm_reduce_scatter PUBLIC
    ${PROJECT_SOURCE_DIR})
  target_compile_options(baseline_cuda_nvshmem_gemm_reduce_scatter PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  add_library(megacu_gemm_reduce_scatter_cuda_native OBJECT
    megacu/gemm_reduce_scatter_megacu.cu)
  target_link_libraries(megacu_gemm_reduce_scatter_cuda_native PUBLIC
    megacu_headers
    CUDA::cudart)
  target_include_directories(megacu_gemm_reduce_scatter_cuda_native PUBLIC
    ${PROJECT_SOURCE_DIR})
  target_compile_options(megacu_gemm_reduce_scatter_cuda_native PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(
    golden_cuda_nvshmem_gemm_reduce_scatter
    baseline_cuda_nvshmem_gemm_reduce_scatter
    megacu_gemm_reduce_scatter_cuda_native
    PROPERTIES
      CUDA_ARCHITECTURES 80
      CUDA_SEPARABLE_COMPILATION ON
      ${MEGACU_GEMM_RS_CUDA_STANDARD_PROPERTIES})

  if(MEGACU_ENABLE_NVSHMEM_TESTS)
    target_compile_definitions(megacu_gemm_reduce_scatter_cuda_native PRIVATE
      MEGACU_GEMM_RS_HAS_DEVICE_NVSHMEM=1
      MEGACU_HAS_DEVICE_NVSHMEM=1)
    target_include_directories(megacu_gemm_reduce_scatter_cuda_native PUBLIC
      ${NVSHMEM_INCLUDE_DIR})
    target_link_libraries(megacu_gemm_reduce_scatter_cuda_native PUBLIC
      CUDA::cuda_driver
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
  endif()
endif()

if(MEGACU_HAS_CUDA_LANGUAGE)
  megacu_add_orchestrate_target(
    TARGET cuda_nvshmem_gemm_reduce_scatter_megacu
    SOURCES megacu/gemm_reduce_scatter_orchestrate.cc
    COMPONENTS cuda_nvshmem_static
    PROGRESS asap)

  target_sources(cuda_nvshmem_gemm_reduce_scatter_megacu PRIVATE
    $<TARGET_OBJECTS:megacu_gemm_reduce_scatter_cuda_native>)
  target_link_libraries(cuda_nvshmem_gemm_reduce_scatter_megacu PUBLIC
    CUDA::cudart)
  set_target_properties(cuda_nvshmem_gemm_reduce_scatter_megacu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
  if(MEGACU_ENABLE_NVSHMEM_TESTS)
    target_link_libraries(cuda_nvshmem_gemm_reduce_scatter_megacu PUBLIC
      CUDA::cuda_driver
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
  endif()

  add_executable(megacu_cuda_gemm_reduce_scatter_correctness
    ${PROJECT_SOURCE_DIR}/tests/runtime/cuda_gemm_reduce_scatter_correctness.cc)
  target_link_libraries(megacu_cuda_gemm_reduce_scatter_correctness PRIVATE
    megacu_headers
    golden_cuda_nvshmem_gemm_reduce_scatter
    baseline_cuda_nvshmem_gemm_reduce_scatter
    cuda_nvshmem_gemm_reduce_scatter_megacu
    CUDA::cudart)
  set_target_properties(megacu_cuda_gemm_reduce_scatter_correctness PROPERTIES
    LINKER_LANGUAGE CUDA)

  add_test(
    NAME cuda_gemm_reduce_scatter_correctness
    COMMAND megacu_cuda_gemm_reduce_scatter_correctness)

  if(MEGACU_ENABLE_NVSHMEM_TESTS)
    add_executable(megacu_nvshmem_two_rank_gemm_rs_smoke
      ${PROJECT_SOURCE_DIR}/tests/runtime/nvshmem_two_rank_gemm_rs_smoke.cc)
    target_link_libraries(megacu_nvshmem_two_rank_gemm_rs_smoke PRIVATE
      megacu_headers
      cuda_nvshmem_gemm_reduce_scatter_megacu
      CUDA::cudart
      CUDA::cuda_driver
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
    target_include_directories(megacu_nvshmem_two_rank_gemm_rs_smoke PRIVATE
      ${NVSHMEM_INCLUDE_DIR})
    set_target_properties(megacu_nvshmem_two_rank_gemm_rs_smoke PROPERTIES
      LINKER_LANGUAGE CUDA)

    foreach(mode
        megacu_host_orch
        megacu_seeded_orch)
      if(mode STREQUAL "megacu_host_orch")
        set(test_name nvshmem_two_rank_gemm_rs_correctness_megacu_host_orch)
      else()
        set(test_name nvshmem_two_rank_gemm_rs_correctness_megacu_seeded_orch)
      endif()
      add_test(
        NAME ${test_name}
        COMMAND ${NVSHMEMRUN_EXECUTABLE} -np 2
                $<TARGET_FILE:megacu_nvshmem_two_rank_gemm_rs_smoke> ${mode})
      set_tests_properties(${test_name} PROPERTIES
        SKIP_RETURN_CODE 77)
    endforeach()

    add_executable(megacu_mpi_two_rank_gemm_rs_smoke
      ${PROJECT_SOURCE_DIR}/tests/runtime/nvshmem_two_rank_gemm_rs_smoke.cc)
    target_compile_definitions(megacu_mpi_two_rank_gemm_rs_smoke PRIVATE
      MEGACU_GEMM_RS_SMOKE_USE_MPI_BOOTSTRAP=1)
    target_link_libraries(megacu_mpi_two_rank_gemm_rs_smoke PRIVATE
      megacu_headers
      cuda_nvshmem_gemm_reduce_scatter_megacu
      CUDA::cudart
      CUDA::cuda_driver
      MPI::MPI_CXX
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
    target_include_directories(megacu_mpi_two_rank_gemm_rs_smoke PRIVATE
      ${NVSHMEM_INCLUDE_DIR})
    set_target_properties(megacu_mpi_two_rank_gemm_rs_smoke PROPERTIES
      LINKER_LANGUAGE CUDA)

    foreach(mode
        megacu_host_orch
        megacu_seeded_orch)
      if(mode STREQUAL "megacu_host_orch")
        set(test_name mpi_two_rank_gemm_rs_correctness_megacu_host_orch)
      else()
        set(test_name mpi_two_rank_gemm_rs_correctness_megacu_seeded_orch)
      endif()
      add_test(
        NAME ${test_name}
        COMMAND mpirun -np 2
                $<TARGET_FILE:megacu_mpi_two_rank_gemm_rs_smoke> ${mode})
      set_tests_properties(${test_name} PROPERTIES
        SKIP_RETURN_CODE 77)
    endforeach()

    add_executable(megacu_torch_uid_two_rank_gemm_rs_smoke
      ${PROJECT_SOURCE_DIR}/tests/runtime/nvshmem_two_rank_gemm_rs_smoke.cc)
    target_compile_definitions(megacu_torch_uid_two_rank_gemm_rs_smoke PRIVATE
      MEGACU_GEMM_RS_SMOKE_USE_UID_BOOTSTRAP=1)
    target_link_libraries(megacu_torch_uid_two_rank_gemm_rs_smoke PRIVATE
      megacu_headers
      cuda_nvshmem_gemm_reduce_scatter_megacu
      CUDA::cudart
      CUDA::cuda_driver
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
    target_include_directories(megacu_torch_uid_two_rank_gemm_rs_smoke PRIVATE
      ${NVSHMEM_INCLUDE_DIR})
    set_target_properties(megacu_torch_uid_two_rank_gemm_rs_smoke PROPERTIES
      LINKER_LANGUAGE CUDA)
  endif()
endif()
