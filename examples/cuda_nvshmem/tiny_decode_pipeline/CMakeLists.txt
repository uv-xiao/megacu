add_library(cuda_nvshmem_tiny_decode_golden STATIC
  golden/tiny_decode_golden.cc)
target_link_libraries(cuda_nvshmem_tiny_decode_golden PUBLIC
  megacu_headers)

if(MEGACU_HAS_CUDA_LANGUAGE)
  add_library(cuda_nvshmem_tiny_decode_baseline STATIC
    baseline/tiny_decode_baseline.cu)
  target_link_libraries(cuda_nvshmem_tiny_decode_baseline PUBLIC
    megacu_headers
    CUDA::cudart)

  add_library(cuda_nvshmem_tiny_decode_megacu STATIC
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

  if(CMAKE_VERSION VERSION_LESS 3.25)
    target_compile_options(cuda_nvshmem_tiny_decode_baseline PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
    target_compile_options(cuda_nvshmem_tiny_decode_megacu PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
  else()
    set_target_properties(cuda_nvshmem_tiny_decode_baseline PROPERTIES
      CUDA_ARCHITECTURES 80
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON)
    set_target_properties(cuda_nvshmem_tiny_decode_megacu PROPERTIES
      CUDA_ARCHITECTURES 80
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON)
  endif()

  add_test(
    NAME tiny_decode_correctness
    COMMAND cuda_nvshmem_tiny_decode_correctness)
endif()
