function(megacu_add_components)
  set(one_value_args NAME DISPATCHER SCHEDULER PLATFORM BACKEND)
  cmake_parse_arguments(MEGACU "" "${one_value_args}" "" ${ARGN})

  if(NOT MEGACU_NAME)
    message(FATAL_ERROR "megacu_add_components requires NAME")
  endif()
  foreach(required DISPATCHER SCHEDULER PLATFORM BACKEND)
    if(NOT MEGACU_${required})
      message(FATAL_ERROR "megacu_add_components requires ${required}")
    endif()
  endforeach()

  add_library(${MEGACU_NAME} STATIC
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/dispatcher/explicit_attrs.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/scheduler/explicit_asap.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/platform/cuda/platform.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/backends/nvshmem/backend.cc
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/target/runtime.cc)
  target_link_libraries(${MEGACU_NAME} PUBLIC megacu_headers)
  target_compile_definitions(${MEGACU_NAME} PRIVATE
    "MEGACU_COMPONENT_DISPATCHER=\"${MEGACU_DISPATCHER}\""
    "MEGACU_COMPONENT_SCHEDULER=\"${MEGACU_SCHEDULER}\""
    "MEGACU_COMPONENT_PLATFORM=\"${MEGACU_PLATFORM}\""
    "MEGACU_COMPONENT_BACKEND=\"${MEGACU_BACKEND}\"")
  set_target_properties(${MEGACU_NAME} PROPERTIES
    MEGACU_DISPATCHER "${MEGACU_DISPATCHER}"
    MEGACU_SCHEDULER "${MEGACU_SCHEDULER}"
    MEGACU_PLATFORM "${MEGACU_PLATFORM}"
    MEGACU_BACKEND "${MEGACU_BACKEND}")
endfunction()

function(megacu_add_orchestrate_target)
  set(one_value_args TARGET COMPONENTS PROGRESS)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(MEGACU "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT MEGACU_TARGET)
    message(FATAL_ERROR "megacu_add_orchestrate_target requires TARGET")
  endif()
  if(NOT MEGACU_COMPONENTS)
    message(FATAL_ERROR "megacu_add_orchestrate_target requires COMPONENTS")
  endif()
  if(NOT MEGACU_SOURCES)
    message(FATAL_ERROR "megacu_add_orchestrate_target requires SOURCES")
  endif()

  add_library(${MEGACU_TARGET} STATIC ${MEGACU_SOURCES})
  target_link_libraries(${MEGACU_TARGET} PUBLIC megacu_headers)

  if(MEGACU_COMPONENTS)
    target_link_libraries(${MEGACU_TARGET} PUBLIC ${MEGACU_COMPONENTS})
  endif()

  target_compile_definitions(${MEGACU_TARGET} PRIVATE
    "MEGACU_TARGET_NAME=\"${MEGACU_TARGET}\""
    "MEGACU_PROGRESS_MODEL=\"${MEGACU_PROGRESS}\"")
  set_target_properties(${MEGACU_TARGET} PROPERTIES
    MEGACU_COMPONENTS "${MEGACU_COMPONENTS}"
    MEGACU_PROGRESS "${MEGACU_PROGRESS}")
endfunction()
