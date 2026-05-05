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
