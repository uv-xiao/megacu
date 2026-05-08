function(megacu_require_example_layout test_name example_dir megacu_source)
  add_test(
    NAME ${test_name}
    COMMAND ${CMAKE_COMMAND} -E env
            bash -lc
            "test -f ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/${example_dir}/README.md && test -f ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/${example_dir}/${megacu_source}")
endfunction()

megacu_require_example_layout(
  gemm_rs_example_layout
  gemm_reduce_scatter
  megacu/gemm_reduce_scatter_megacu.cu)
megacu_require_example_layout(
  ag_gemm_example_layout
  allgather_gemm
  megacu/allgather_gemm_megacu.cu)
megacu_require_example_layout(
  tiny_decode_example_layout
  tiny_decode_pipeline
  megacu/tiny_decode_megacu.cu)

add_subdirectory(gemm_allreduce)
add_subdirectory(gemm_reduce_scatter)
add_subdirectory(allgather_gemm)
add_subdirectory(tiny_decode_pipeline)

add_executable(megacu_runtime_distributed_recipe_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_distributed_recipe_contracts.cc)
target_link_libraries(megacu_runtime_distributed_recipe_contracts PRIVATE
  megacu_headers)

add_test(
  NAME runtime_distributed_recipe_contracts
  COMMAND megacu_runtime_distributed_recipe_contracts)

add_test(
  NAME examples_do_not_define_host_orch_execution_model
  COMMAND bash -lc
          "! grep -R --include='*.cu' --include='*.cuh' \"struct host_orch_execution\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem")

add_test(
  NAME tiny_decode_uses_common_local_event_tensor
  COMMAND bash -lc
          "grep -R \"struct local_event_tensor_i32\" ${PROJECT_SOURCE_DIR}/include/megacu/platform/cuda && ! grep -R --include='*.cu' --include='*.cuh' \"struct local_event_tensor_i32\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem")

add_test(
  NAME mpi_runner_covers_gemm_rs_and_ag_gemm
  COMMAND bash -lc
          "grep -q \"megacu_mpi_two_rank_gemm_rs_smoke\" ${PROJECT_SOURCE_DIR}/tools/cuda_nvshmem/run_mpi.sh && grep -q \"megacu_mpi_two_rank_ag_gemm_smoke\" ${PROJECT_SOURCE_DIR}/tools/cuda_nvshmem/run_mpi.sh && grep -q \"gemm_rs\" ${PROJECT_SOURCE_DIR}/tools/cuda_nvshmem/run_mpi.sh && grep -q \"ag_gemm\" ${PROJECT_SOURCE_DIR}/tools/cuda_nvshmem/run_mpi.sh")

add_test(
  NAME torch_runner_covers_gemm_rs_and_ag_gemm
  COMMAND bash -lc
          "grep -q \"megacu_torch_uid_two_rank_gemm_rs_smoke\" ${PROJECT_SOURCE_DIR}/tools/cuda_nvshmem/run_torch.py && grep -q \"megacu_torch_uid_two_rank_ag_gemm_smoke\" ${PROJECT_SOURCE_DIR}/tools/cuda_nvshmem/run_torch.py")
