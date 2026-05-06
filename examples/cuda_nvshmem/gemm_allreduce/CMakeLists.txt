megacu_add_components(
  NAME cuda_nvshmem_static
  DISPATCHER explicit_attrs
  SCHEDULER explicit_asap
  PLATFORM cuda
  BACKEND nvshmem)

if(MEGACU_ENABLE_NVSHMEM_TESTS)
  if(NOT MEGACU_HAS_CUDA_LANGUAGE)
    message(FATAL_ERROR
      "MEGACU_ENABLE_NVSHMEM_TESTS requires a CUDA compiler for numeric kernels")
  endif()

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
  add_library(golden_cuda_nvshmem_gemm_allreduce STATIC
    golden/golden_gemm_allreduce.cu)
  target_link_libraries(golden_cuda_nvshmem_gemm_allreduce PUBLIC
    CUDA::cudart)
  target_include_directories(golden_cuda_nvshmem_gemm_allreduce PUBLIC
    ${PROJECT_SOURCE_DIR})
  if(CMAKE_VERSION VERSION_LESS 3.25)
    target_compile_options(golden_cuda_nvshmem_gemm_allreduce PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
    set(MEGACU_GEMM_AR_CUDA_STANDARD_PROPERTIES "")
  else()
    set(MEGACU_GEMM_AR_CUDA_STANDARD_PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON)
  endif()
  set_target_properties(golden_cuda_nvshmem_gemm_allreduce PROPERTIES
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
    ${MEGACU_GEMM_AR_CUDA_STANDARD_PROPERTIES})

  if(MEGACU_ENABLE_NVSHMEM_TESTS)
    target_compile_definitions(golden_cuda_nvshmem_gemm_allreduce PRIVATE
      MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM=1)
    target_include_directories(golden_cuda_nvshmem_gemm_allreduce PUBLIC
      ${NVSHMEM_INCLUDE_DIR})
    target_link_libraries(golden_cuda_nvshmem_gemm_allreduce PUBLIC
      CUDA::cuda_driver
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
  endif()

  add_library(manual_megakernel_cuda_nvshmem_gemm_allreduce OBJECT
    phased/baseline/manual_megakernel_gemm_allreduce.cu)
  target_link_libraries(manual_megakernel_cuda_nvshmem_gemm_allreduce PUBLIC
    megacu_headers
    CUDA::cudart)
  target_include_directories(manual_megakernel_cuda_nvshmem_gemm_allreduce PUBLIC
    ${PROJECT_SOURCE_DIR})
  if(CMAKE_VERSION VERSION_LESS 3.25)
    target_compile_options(manual_megakernel_cuda_nvshmem_gemm_allreduce PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
  endif()
  set_target_properties(manual_megakernel_cuda_nvshmem_gemm_allreduce PROPERTIES
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
    ${MEGACU_GEMM_AR_CUDA_STANDARD_PROPERTIES})

  if(MEGACU_ENABLE_NVSHMEM_TESTS)
    target_compile_definitions(manual_megakernel_cuda_nvshmem_gemm_allreduce PRIVATE
      MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM=1
      MEGACU_HAS_DEVICE_NVSHMEM=1)
    target_include_directories(manual_megakernel_cuda_nvshmem_gemm_allreduce PUBLIC
      ${NVSHMEM_INCLUDE_DIR})
    target_link_libraries(manual_megakernel_cuda_nvshmem_gemm_allreduce PUBLIC
      CUDA::cuda_driver
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
  endif()

  add_library(megacu_gemm_allreduce_cuda_native OBJECT
    phased/megacu/megacu_gemm_allreduce_phased.cu)
  target_link_libraries(megacu_gemm_allreduce_cuda_native PUBLIC
    megacu_headers
    CUDA::cudart)
  if(CMAKE_VERSION VERSION_LESS 3.25)
    target_compile_options(megacu_gemm_allreduce_cuda_native PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
  endif()
  set_target_properties(megacu_gemm_allreduce_cuda_native PROPERTIES
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
    ${MEGACU_GEMM_AR_CUDA_STANDARD_PROPERTIES})

  if(MEGACU_ENABLE_NVSHMEM_TESTS)
    target_compile_definitions(megacu_gemm_allreduce_cuda_native PRIVATE
      MEGACU_GEMM_AR_HAS_DEVICE_NVSHMEM=1
      MEGACU_HAS_DEVICE_NVSHMEM=1)
    target_include_directories(megacu_gemm_allreduce_cuda_native PUBLIC
      ${NVSHMEM_INCLUDE_DIR})
    target_link_libraries(megacu_gemm_allreduce_cuda_native PUBLIC
      CUDA::cuda_driver
      ${NVSHMEM_DEVICE_LIBRARY}
      ${NVSHMEM_HOST_LIBRARY})
  endif()

endif()

add_subdirectory(phased/megacu)

add_executable(megacu_runtime_linked_surface_contract
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_linked_surface_contract.cc)
target_link_libraries(megacu_runtime_linked_surface_contract PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_linked_surface_contract
  COMMAND megacu_runtime_linked_surface_contract)

add_executable(megacu_runtime_components_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_components_contracts.cc)
target_link_libraries(megacu_runtime_components_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_components_contracts
  COMMAND megacu_runtime_components_contracts)

add_executable(megacu_runtime_execution_models_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_execution_models_contracts.cc)
target_link_libraries(megacu_runtime_execution_models_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_execution_models_contracts
  COMMAND megacu_runtime_execution_models_contracts)

add_executable(megacu_runtime_arena_execution_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_arena_execution_contracts.cc)
target_link_libraries(megacu_runtime_arena_execution_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_arena_execution_contracts
  COMMAND megacu_runtime_arena_execution_contracts)

add_executable(megacu_runtime_strategy_selection_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_strategy_selection_contracts.cc)
target_link_libraries(megacu_runtime_strategy_selection_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_strategy_selection_contracts
  COMMAND megacu_runtime_strategy_selection_contracts)

add_executable(megacu_runtime_gemm_allreduce_recipe_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_gemm_allreduce_recipe_contracts.cc)
target_link_libraries(megacu_runtime_gemm_allreduce_recipe_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_gemm_allreduce_recipe_contracts
  COMMAND megacu_runtime_gemm_allreduce_recipe_contracts)

add_executable(megacu_runtime_gemm_allreduce_host_arena_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/runtime_gemm_allreduce_host_arena_contracts.cc)
target_link_libraries(megacu_runtime_gemm_allreduce_host_arena_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME runtime_gemm_allreduce_host_arena_contracts
  COMMAND megacu_runtime_gemm_allreduce_host_arena_contracts)

add_executable(megacu_adapter_contracts
  ${PROJECT_SOURCE_DIR}/tests/build/adapter_contracts.cc)
target_link_libraries(megacu_adapter_contracts PRIVATE
  megacu_headers
  cuda_nvshmem_static)

add_test(
  NAME adapter_local_shape_contract
  COMMAND megacu_adapter_contracts)

add_test(
  NAME mpi_adapter_env_contract
  COMMAND bash -lc
          "MEGACU_ADAPTER_CONTRACT_MODE=mpi OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_LOCAL_RANK=1 $<TARGET_FILE:megacu_adapter_contracts>")

add_test(
  NAME mpi_adapter_requires_local_rank
  COMMAND bash -lc
          "! MEGACU_ADAPTER_CONTRACT_MODE=mpi OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_SIZE=2 $<TARGET_FILE:megacu_adapter_contracts>")

add_test(
  NAME torch_adapter_env_contract
  COMMAND bash -lc
          "MEGACU_ADAPTER_CONTRACT_MODE=torch RANK=1 WORLD_SIZE=2 LOCAL_RANK=1 $<TARGET_FILE:megacu_adapter_contracts>")

add_test(
  NAME cuda_nvshmem_script_layout
  COMMAND bash -lc
          "test -x ${PROJECT_SOURCE_DIR}/tools/cuda_nvshmem/run_direct.sh && test -x ${PROJECT_SOURCE_DIR}/tools/cuda_nvshmem/run_mpi.sh && test -x ${PROJECT_SOURCE_DIR}/tools/cuda_nvshmem/run_torch.py")

if(MEGACU_HAS_CUDA_LANGUAGE)
  add_executable(megacu_runtime_device_persistent_contract
    ${PROJECT_SOURCE_DIR}/tests/build/runtime_device_persistent_contract.cu)
  target_link_libraries(megacu_runtime_device_persistent_contract PRIVATE
    megacu_headers
    CUDA::cudart)
  target_compile_options(megacu_runtime_device_persistent_contract PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  add_test(
    NAME runtime_device_persistent_contract
    COMMAND megacu_runtime_device_persistent_contract)

  add_executable(megacu_runtime_arena_execution_contract
    ${PROJECT_SOURCE_DIR}/tests/build/runtime_arena_execution_contract.cu)
  target_link_libraries(megacu_runtime_arena_execution_contract PRIVATE
    megacu_headers
    CUDA::cudart)
  target_compile_options(megacu_runtime_arena_execution_contract PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  add_test(
    NAME runtime_arena_execution_contract
    COMMAND megacu_runtime_arena_execution_contract)

  add_executable(megacu_runtime_gemm_allreduce_recipe_device_contract
    ${PROJECT_SOURCE_DIR}/tests/build/runtime_gemm_allreduce_recipe_device_contract.cu)
  target_link_libraries(megacu_runtime_gemm_allreduce_recipe_device_contract PRIVATE
    megacu_headers
    CUDA::cudart)
  target_compile_options(megacu_runtime_gemm_allreduce_recipe_device_contract PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  add_test(
    NAME runtime_gemm_allreduce_recipe_device_contract
    COMMAND megacu_runtime_gemm_allreduce_recipe_device_contract)
endif()

add_test(
  NAME megacu_native_does_not_call_golden
  COMMAND bash -lc
          "! grep -R \"golden_\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/megacu")

add_test(
  NAME megacu_phased_path_uses_single_native_launcher
  COMMAND bash -lc
          "! grep -R \"gemm_tile_produce_kernel<<<\\|allreduce_tile_consume_kernel<<<\\|nvshmem_allreduce_tile_consume_kernel<<<\\|megacu_local_gemm_allreduce_kernel\\|megacu_gemm_stage_kernel\\|megacu_allreduce_stage_kernel\\|manual_gemm_allreduce_megakernel\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/megacu")

add_test(
  NAME manual_megakernel_lives_in_baseline
  COMMAND bash -lc
          "grep -R \"manual_gemm_allreduce_megakernel\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/baseline && ! grep -R \"manual_gemm_allreduce_megakernel\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/megacu")

add_test(
  NAME cuda_megakernel_shell_is_general_platform_code
  COMMAND bash -lc
          "grep -R \"launch_megakernel\" ${PROJECT_SOURCE_DIR}/include/megacu/platform/cuda && ! grep -R \"asap_tile_composer\\|cuda_nvshmem_asap\" ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/megacu")

add_test(
  NAME megakernel_running_logic_is_common
  COMMAND bash -lc
          "grep -R \"struct block_tile_runtime\" ${PROJECT_SOURCE_DIR}/include/megacu/runtime && grep -R \"struct explicit_asap\" ${PROJECT_SOURCE_DIR}/include/megacu/scheduler && grep -R \"struct tile_grid\" ${PROJECT_SOURCE_DIR}/include/megacu/dispatcher && ! grep -R \"gemm_allreduce_program\\|runtime::device::entry\\|device_entry.cuh\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/megacu ${PROJECT_SOURCE_DIR}/include/megacu/runtime")

add_test(
  NAME nvshmem_event_tensor_is_backend_code
  COMMAND bash -lc
          "grep -R \"event_tensor_i32\" ${PROJECT_SOURCE_DIR}/include/megacu/backends/nvshmem && ! grep -R \"struct event_tensor_i32\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/megacu")

add_test(
  NAME megacu_asap_scheduler_has_no_phase_attrs
  COMMAND bash -lc
          "! grep -R --include='*.h' --include='*.cc' --include='*.cu' \"scheduler::phase\\|scheduler_phase\\|attr_kind::scheduler_phase\" ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce ${PROJECT_SOURCE_DIR}/tests/build")

add_test(
  NAME megacu_event_tensor_is_not_scheduler_attr
  COMMAND bash -lc
          "! grep -R --include='*.h' --include='*.cc' --include='*.cu' \"scheduler::event_shape\\|scheduler_event_shape\" ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce ${PROJECT_SOURCE_DIR}/tests/build")

add_test(
  NAME runtime_strategy_selection_contracts_examples_do_not_call_scheduler_run
  COMMAND bash -lc
          "! grep -R --include='*.h' --include='*.cc' --include='*.cu' --include='*.cuh' \"scheduler\\.run\" ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/common ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/megacu")

add_test(
  NAME megacu_event_tensor_uses_decoupled_api
  COMMAND bash -lc
          "! grep -R --include='*.h' --include='*.cc' --include='*.cu' \"runtime::events\\|cuda_nvshmem::events\\|event_publish\\|event_acquire\\|event_join\\|event_publish_attrs\\|event_acquire_attrs\\|event_join_attrs\" ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/common ${PROJECT_SOURCE_DIR}/examples/cuda_nvshmem/gemm_allreduce/phased/megacu ${PROJECT_SOURCE_DIR}/tests/build")

add_executable(megacu_direct_orchestrate_smoke
  ${PROJECT_SOURCE_DIR}/tests/runtime/direct_orchestrate_smoke.cc)
target_link_libraries(megacu_direct_orchestrate_smoke PRIVATE
  megacu_headers
  cuda_nvshmem_gemm_allreduce_phased)
if(MEGACU_HAS_CUDA_LANGUAGE)
  set_target_properties(megacu_direct_orchestrate_smoke PROPERTIES
    LINKER_LANGUAGE CUDA)
endif()

add_test(
  NAME direct_orchestrate_smoke
  COMMAND megacu_direct_orchestrate_smoke)

if(CUDAToolkit_FOUND)
  add_executable(megacu_cuda_orchestrate_smoke
    ${PROJECT_SOURCE_DIR}/tests/runtime/cuda_orchestrate_smoke.cc)
  target_link_libraries(megacu_cuda_orchestrate_smoke PRIVATE
    megacu_headers
    cuda_nvshmem_gemm_allreduce_phased
    CUDA::cudart)
  set_target_properties(megacu_cuda_orchestrate_smoke PROPERTIES
    LINKER_LANGUAGE CUDA)

  add_test(
    NAME cuda_orchestrate_smoke
    COMMAND megacu_cuda_orchestrate_smoke)

  add_executable(megacu_cuda_multicard_smoke
    ${PROJECT_SOURCE_DIR}/tests/runtime/cuda_multicard_smoke.cc)
  target_link_libraries(megacu_cuda_multicard_smoke PRIVATE
    megacu_headers
    cuda_nvshmem_gemm_allreduce_phased
    CUDA::cudart)
  set_target_properties(megacu_cuda_multicard_smoke PROPERTIES
    LINKER_LANGUAGE CUDA)

  add_test(
    NAME cuda_multicard_smoke
    COMMAND megacu_cuda_multicard_smoke)
  set_tests_properties(cuda_multicard_smoke PROPERTIES
    SKIP_RETURN_CODE 77)

  if(MEGACU_HAS_CUDA_LANGUAGE)
    add_executable(megacu_cuda_gemm_allreduce_correctness
      ${PROJECT_SOURCE_DIR}/tests/runtime/cuda_gemm_allreduce_correctness.cc)
    target_link_libraries(megacu_cuda_gemm_allreduce_correctness PRIVATE
      megacu_headers
      golden_cuda_nvshmem_gemm_allreduce
      cuda_nvshmem_gemm_allreduce_phased
      CUDA::cudart)
    set_target_properties(megacu_cuda_gemm_allreduce_correctness PROPERTIES
      LINKER_LANGUAGE CUDA)

    add_test(
      NAME cuda_gemm_allreduce_correctness
      COMMAND megacu_cuda_gemm_allreduce_correctness)
  endif()
endif()

if(MEGACU_ENABLE_NVSHMEM_TESTS)
  add_executable(megacu_nvshmem_two_rank_smoke
    ${PROJECT_SOURCE_DIR}/tests/runtime/nvshmem_two_rank_smoke.cc)
  target_link_libraries(megacu_nvshmem_two_rank_smoke PRIVATE
    megacu_headers
    golden_cuda_nvshmem_gemm_allreduce
    cuda_nvshmem_gemm_allreduce_phased
    CUDA::cudart
    CUDA::cuda_driver
    ${NVSHMEM_DEVICE_LIBRARY}
    ${NVSHMEM_HOST_LIBRARY})
  set_target_properties(megacu_nvshmem_two_rank_smoke PROPERTIES
    LINKER_LANGUAGE CUDA)

  foreach(mode
      golden_phased
      megacu_phased)
    if(mode STREQUAL "golden_phased")
      set(test_name nvshmem_two_rank_gemm_ar_correctness_golden)
    else()
      set(test_name nvshmem_two_rank_gemm_ar_correctness_megacu)
    endif()
    add_test(
      NAME ${test_name}
      COMMAND ${NVSHMEMRUN_EXECUTABLE} -np 2
              $<TARGET_FILE:megacu_nvshmem_two_rank_smoke> ${mode})
    set_tests_properties(${test_name} PROPERTIES
      SKIP_RETURN_CODE 77)
  endforeach()
endif()
