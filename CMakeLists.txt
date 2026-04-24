cmake_minimum_required(VERSION 3.20)

project(megacu LANGUAGES CXX)

add_library(megacu_headers INTERFACE)
target_include_directories(megacu_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(megacu_headers INTERFACE cxx_std_20)

enable_testing()

add_executable(megacu_compile_gemm_allreduce_descriptor
  tests/build/compile_gemm_allreduce_descriptor.cc)
target_link_libraries(megacu_compile_gemm_allreduce_descriptor PRIVATE
  megacu_headers)

add_test(
  NAME compile_gemm_allreduce_descriptor
  COMMAND megacu_compile_gemm_allreduce_descriptor)
