function(megacu_add_components)
  set(one_value_args NAME DISPATCHER SCHEDULER KERNEL_LOWERING PLATFORM BACKEND)
  cmake_parse_arguments(MEGACU "" "${one_value_args}" "" ${ARGN})

  if(NOT MEGACU_NAME)
    message(FATAL_ERROR "megacu_add_components requires NAME")
  endif()
  foreach(required DISPATCHER SCHEDULER KERNEL_LOWERING PLATFORM BACKEND)
    if(NOT MEGACU_${required})
      message(FATAL_ERROR "megacu_add_components requires ${required}")
    endif()
  endforeach()

  add_library(${MEGACU_NAME} STATIC
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/program/materialize.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/dispatcher/tiled_compute_comm_dispatch.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/scheduler/static_persistent.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/lowering/persistent_stitch.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/platform/cuda/platform.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/platform/cuda/validation.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/backends/nvshmem/backend.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/backends/nvshmem/validation.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/target/runtime.cc)
  target_compile_features(${MEGACU_NAME} PUBLIC cxx_std_20)
  target_link_libraries(${MEGACU_NAME} PUBLIC megacu_headers)
  target_compile_definitions(${MEGACU_NAME} PRIVATE
    "MEGACU_COMPONENT_DISPATCHER=\"${MEGACU_DISPATCHER}\""
    "MEGACU_COMPONENT_SCHEDULER=\"${MEGACU_SCHEDULER}\""
    "MEGACU_COMPONENT_KERNEL_LOWERING=\"${MEGACU_KERNEL_LOWERING}\""
    "MEGACU_COMPONENT_PLATFORM=\"${MEGACU_PLATFORM}\""
    "MEGACU_COMPONENT_BACKEND=\"${MEGACU_BACKEND}\"")
  set_target_properties(${MEGACU_NAME} PROPERTIES
    MEGACU_DISPATCHER "${MEGACU_DISPATCHER}"
    MEGACU_SCHEDULER "${MEGACU_SCHEDULER}"
    MEGACU_KERNEL_LOWERING "${MEGACU_KERNEL_LOWERING}"
    MEGACU_PLATFORM "${MEGACU_PLATFORM}"
    MEGACU_BACKEND "${MEGACU_BACKEND}")
endfunction()

function(megacu_add_orchestrate_target)
  set(one_value_args TARGET PROGRAM COMPONENTS SCHEDULER_MODE)
  set(multi_value_args SOURCES BACKEND_ENVELOPE OPS KERNELS)
  cmake_parse_arguments(MEGACU "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT MEGACU_TARGET)
    message(FATAL_ERROR "megacu_add_orchestrate_target requires TARGET")
  endif()
  if(NOT MEGACU_PROGRAM)
    message(FATAL_ERROR "megacu_add_orchestrate_target requires PROGRAM")
  endif()
  if(NOT MEGACU_COMPONENTS)
    message(FATAL_ERROR "megacu_add_orchestrate_target requires COMPONENTS")
  endif()
  if(NOT MEGACU_SOURCES)
    message(FATAL_ERROR "megacu_add_orchestrate_target requires SOURCES")
  endif()

  add_library(${MEGACU_TARGET} STATIC ${MEGACU_SOURCES})
  target_compile_features(${MEGACU_TARGET} PUBLIC cxx_std_20)
  target_link_libraries(${MEGACU_TARGET} PUBLIC megacu_headers)

  if(MEGACU_COMPONENTS)
    target_link_libraries(${MEGACU_TARGET} PUBLIC ${MEGACU_COMPONENTS})
  endif()

  target_compile_definitions(${MEGACU_TARGET} PRIVATE
    "MEGACU_TARGET_NAME=\"${MEGACU_TARGET}\""
    "MEGACU_PROGRAM_NAME=\"${MEGACU_PROGRAM}\""
    "MEGACU_SCHEDULER_MODE=\"${MEGACU_SCHEDULER_MODE}\"")
  set_target_properties(${MEGACU_TARGET} PROPERTIES
    MEGACU_PROGRAM "${MEGACU_PROGRAM}"
    MEGACU_COMPONENTS "${MEGACU_COMPONENTS}"
    MEGACU_SCHEDULER_MODE "${MEGACU_SCHEDULER_MODE}")
endfunction()
