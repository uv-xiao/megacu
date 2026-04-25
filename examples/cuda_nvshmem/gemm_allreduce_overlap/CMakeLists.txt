megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_overlap
  PROGRAM gemm_allreduce_overlap_program
  SOURCES gemm_allreduce_overlap_orchestrate.cc
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE co_resident_persistent
  BACKEND_ENVELOPE TEAM_SIZE 2)

if(MEGACU_HAS_CUDA_LANGUAGE)
  target_sources(cuda_nvshmem_gemm_allreduce_overlap PRIVATE
    $<TARGET_OBJECTS:megacu_gemm_allreduce_cuda_native>)
  target_link_libraries(cuda_nvshmem_gemm_allreduce_overlap PUBLIC
    golden_cuda_nvshmem_gemm_allreduce
    CUDA::cudart)
  target_compile_definitions(cuda_nvshmem_gemm_allreduce_overlap PRIVATE
    MEGACU_HAS_CUDA_NUMERIC_PATH=1)
endif()
