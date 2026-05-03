megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_phased
  SOURCES gemm_allreduce_phased_orchestrate.cc
  COMPONENTS cuda_nvshmem_static
  PROGRESS asap)

if(MEGACU_HAS_CUDA_LANGUAGE)
  target_sources(cuda_nvshmem_gemm_allreduce_phased PRIVATE
    $<TARGET_OBJECTS:megacu_gemm_allreduce_cuda_native>)
  target_link_libraries(cuda_nvshmem_gemm_allreduce_phased PUBLIC
    CUDA::cudart)
  set_target_properties(cuda_nvshmem_gemm_allreduce_phased PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(cuda_nvshmem_gemm_allreduce_phased PRIVATE
    MEGACU_HAS_CUDA_NUMERIC_PATH=1)
  if(MEGACU_ENABLE_NVSHMEM_TESTS)
    target_link_libraries(cuda_nvshmem_gemm_allreduce_phased PUBLIC
      CUDA::cuda_driver
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
  endif()
endif()
