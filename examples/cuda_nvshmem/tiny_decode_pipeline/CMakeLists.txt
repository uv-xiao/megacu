add_library(cuda_nvshmem_tiny_decode_golden STATIC
  golden/tiny_decode_golden.cc)
target_link_libraries(cuda_nvshmem_tiny_decode_golden PUBLIC
  megacu_headers)

if(MEGACU_ENABLE_NVSHMEM_TESTS)
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
  add_library(cuda_nvshmem_tiny_decode_baseline STATIC
    baseline/tiny_decode_baseline.cu)
  target_link_libraries(cuda_nvshmem_tiny_decode_baseline PUBLIC
    megacu_headers
    CUDA::cudart)

  add_library(cuda_nvshmem_tiny_decode_megacu STATIC
    megacu/tiny_decode_arena.cc
    megacu/tiny_decode_megacu.cu)
  target_link_libraries(cuda_nvshmem_tiny_decode_megacu PUBLIC
    megacu_headers
    cuda_nvshmem_static
    CUDA::cudart)
  target_compile_options(cuda_nvshmem_tiny_decode_megacu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  add_executable(cuda_nvshmem_tiny_decode_correctness
    ${PROJECT_SOURCE_DIR}/tests/runtime/tiny_decode_correctness.cc)
  target_link_libraries(cuda_nvshmem_tiny_decode_correctness PRIVATE
    megacu_headers
    cuda_nvshmem_tiny_decode_golden
    cuda_nvshmem_tiny_decode_baseline
    cuda_nvshmem_tiny_decode_megacu
    CUDA::cudart)
  set_target_properties(cuda_nvshmem_tiny_decode_correctness PROPERTIES
    LINKER_LANGUAGE CUDA)

  set_target_properties(cuda_nvshmem_tiny_decode_baseline PROPERTIES
    CUDA_ARCHITECTURES 80)
  set_target_properties(cuda_nvshmem_tiny_decode_megacu PROPERTIES
    CUDA_ARCHITECTURES 80)

  add_test(
    NAME tiny_decode_correctness
    COMMAND cuda_nvshmem_tiny_decode_correctness)

  if(MEGACU_ENABLE_NVSHMEM_TESTS)
    add_executable(nvshmem_two_rank_tiny_decode_smoke
      ${PROJECT_SOURCE_DIR}/tests/runtime/nvshmem_two_rank_tiny_decode_smoke.cc)
    target_link_libraries(nvshmem_two_rank_tiny_decode_smoke PRIVATE
      megacu_headers
      cuda_nvshmem_tiny_decode_golden
      cuda_nvshmem_tiny_decode_megacu
      CUDA::cudart
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
    target_include_directories(nvshmem_two_rank_tiny_decode_smoke PRIVATE
      ${NVSHMEM_INCLUDE_DIR})
    set_target_properties(nvshmem_two_rank_tiny_decode_smoke PROPERTIES
      LINKER_LANGUAGE CUDA)

    foreach(mode
        megacu_host_orch
        megacu_seeded_orch)
      if(mode STREQUAL "megacu_host_orch")
        set(test_name nvshmem_two_rank_tiny_decode_correctness_megacu_host_orch)
      else()
        set(test_name nvshmem_two_rank_tiny_decode_correctness_megacu_seeded_orch)
      endif()
      add_test(
        NAME ${test_name}
        COMMAND ${NVSHMEMRUN_EXECUTABLE} -np 2
                $<TARGET_FILE:nvshmem_two_rank_tiny_decode_smoke> ${mode})
      set_tests_properties(${test_name} PROPERTIES
        SKIP_RETURN_CODE 77)
    endforeach()
  endif()
endif()
