cmake_minimum_required(VERSION 3.20)

project(megacu LANGUAGES CXX)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++20>)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>)
endif()

add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=3012>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=3013>)

option(MEGACU_ENABLE_NVSHMEM_TESTS
  "Build CUDA+NVSHMEM two-rank runtime validation tests" OFF)

find_package(CUDAToolkit QUIET)
set(MEGACU_HAS_CUDA_LANGUAGE OFF)
if(CUDAToolkit_FOUND)
  include(CheckLanguage)
  if(NOT CMAKE_CUDA_COMPILER)
    find_program(MEGACU_PREFERRED_NVCC
      NAMES nvcc
      PATHS /usr/local/cuda-12.8/bin /usr/local/cuda/bin
      NO_DEFAULT_PATH)
    if(MEGACU_PREFERRED_NVCC)
      set(CMAKE_CUDA_COMPILER "${MEGACU_PREFERRED_NVCC}"
        CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING
        "CUDA architectures built by Megacu examples and tests")
    endif()
    enable_language(CUDA)
    set(MEGACU_HAS_CUDA_LANGUAGE ON)
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MegacuTargets)

add_library(megacu_headers INTERFACE)
target_include_directories(megacu_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

add_subdirectory(examples/cuda_nvshmem)
