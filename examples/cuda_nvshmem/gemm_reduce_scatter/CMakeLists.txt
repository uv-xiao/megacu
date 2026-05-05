if(MEGACU_HAS_CUDA_LANGUAGE)
  add_library(cuda_nvshmem_gemm_reduce_scatter_megacu OBJECT
    megacu/gemm_reduce_scatter_megacu.cu)
  target_link_libraries(cuda_nvshmem_gemm_reduce_scatter_megacu PRIVATE
    megacu_headers)
  target_compile_options(cuda_nvshmem_gemm_reduce_scatter_megacu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  if(CMAKE_VERSION VERSION_LESS 3.25)
    target_compile_options(cuda_nvshmem_gemm_reduce_scatter_megacu PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
  else()
    set_target_properties(cuda_nvshmem_gemm_reduce_scatter_megacu PROPERTIES
      CUDA_ARCHITECTURES 80
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON)
  endif()
endif()
