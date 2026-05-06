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
  if(CMAKE_VERSION VERSION_LESS 3.25)
    target_compile_options(golden_cuda_nvshmem_gemm_reduce_scatter PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
    target_compile_options(baseline_cuda_nvshmem_gemm_reduce_scatter PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
    target_compile_options(megacu_gemm_reduce_scatter_cuda_native PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
  else()
    set(MEGACU_GEMM_RS_CUDA_STANDARD_PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON)
  endif()
  set_target_properties(
    golden_cuda_nvshmem_gemm_reduce_scatter
    baseline_cuda_nvshmem_gemm_reduce_scatter
    megacu_gemm_reduce_scatter_cuda_native
    PROPERTIES
      CUDA_ARCHITECTURES 80
      CUDA_SEPARABLE_COMPILATION ON
      ${MEGACU_GEMM_RS_CUDA_STANDARD_PROPERTIES})
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
endif()
