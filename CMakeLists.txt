cmake_minimum_required(VERSION 3.20)

project(megacu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    enable_language(CUDA)
    set(MEGACU_HAS_CUDA_LANGUAGE ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MegacuTargets)

add_library(megacu_headers INTERFACE)
target_include_directories(megacu_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(megacu_headers INTERFACE cxx_std_20)

enable_testing()

add_subdirectory(examples/cuda_nvshmem)
