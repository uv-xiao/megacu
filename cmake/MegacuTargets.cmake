function(megacu_add_components)
  set(one_value_args NAME DISPATCHER SCHEDULER KERNEL_LOWERING PLATFORM BACKEND)
  cmake_parse_arguments(MEGACU "" "${one_value_args}" "" ${ARGN})

  if(NOT MEGACU_NAME)
    message(FATAL_ERROR "megacu_add_components requires NAME")
  endif()

  add_library(${MEGACU_NAME} INTERFACE)
  target_compile_features(${MEGACU_NAME} INTERFACE cxx_std_20)
  target_link_libraries(${MEGACU_NAME} INTERFACE megacu_headers)
endfunction()

function(megacu_add_orchestrate_target)
  set(one_value_args TARGET PROGRAM COMPONENTS SCHEDULER_MODE)
  set(multi_value_args SOURCES BACKEND_ENVELOPE OPS KERNELS)
  cmake_parse_arguments(MEGACU "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT MEGACU_TARGET)
    message(FATAL_ERROR "megacu_add_orchestrate_target requires TARGET")
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
    "MEGACU_TARGET_NAME=${MEGACU_TARGET}"
    "MEGACU_PROGRAM_NAME=${MEGACU_PROGRAM}"
    "MEGACU_SCHEDULER_MODE=${MEGACU_SCHEDULER_MODE}")
endfunction()
