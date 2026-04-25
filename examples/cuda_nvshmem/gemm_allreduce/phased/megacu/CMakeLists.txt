megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_phased
  PROGRAM gemm_allreduce_phased_program
  SOURCES gemm_allreduce_phased_orchestrate.cc
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE phased
  BACKEND_ENVELOPE TEAM_SIZE 2)

if(MEGACU_HAS_CUDA_LANGUAGE)
  target_sources(cuda_nvshmem_gemm_allreduce_phased PRIVATE
    $<TARGET_OBJECTS:megacu_gemm_allreduce_cuda_native>)
  target_link_libraries(cuda_nvshmem_gemm_allreduce_phased PUBLIC
    golden_cuda_nvshmem_gemm_allreduce
    CUDA::cudart)
  target_compile_definitions(cuda_nvshmem_gemm_allreduce_phased PRIVATE
    MEGACU_HAS_CUDA_NUMERIC_PATH=1)
endif()
