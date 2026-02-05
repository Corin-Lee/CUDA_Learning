cmake_minimum_required(VERSION 3.22)
project(SoftmaxOptimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

set(SOURCES
  src/main.cu
  src/test.cpp

  src/kernels/softmax_cpu_v1.cpp
  src/kernels/softmax_gpu_v1.cu
  src/kernels/softmax_gpu_v2.cu
  src/kernels/softmax_gpu_v3.cu

  src/util/launch_softmax_cpu.cpp
  src/util/launch_softmax_gpu.cu
)

add_executable(softmax_test ${SOURCES})


target_include_directories(softmax_test PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(softmax_test PRIVATE CUDA::cudart)

target_compile_options(softmax_test PRIVATE
$<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
$<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
)