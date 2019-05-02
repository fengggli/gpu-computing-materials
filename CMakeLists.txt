cmake_minimum_required(VERSION 3.5)
#set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(IS_CI_BUILD "Is a build for CI" OFF) # Reduce problem size if it is run in CI
option(USE_CLANG "Use clang compiler set" OFF)
option(USE_OPENBLAS "USE openblas for blas" ON)
option(BUILD_TEST "Build google test and all test cases" ON)
option(USE_CUDA "Build with CUDA" OFF)
option(USE_CUDNN "Build with CUDNN" OFF)
option(USE_NNPACK "Build with NNPACK" OFF)
option(AWNN_USE_FLT32 "Use single precision floating point" OFF)
option(AWNN_USE_OPENMP "Use OpenMP" OFF)
set (AWNN_GEMM_THREADS 1 CACHE STRING "number of openblas threads")

option(USE_STRICT_CHECK "use pedantic flag" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

if(USE_CLANG)
  set(CMAKE_C_COMPILER "clang")
  set(CMAKE_CXX_COMPILER "clang++")
else()
  set(CMAKE_C_COMPILER "gcc")
  set(CMAKE_CXX_COMPILER "g++")
endif()

set(CMAKE_CXX_STANDARD 11)


set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wextra -Wconversion -rdynamic")
if(USE_STRICT_CHECK)
  set(GCC_COVERAGE_COMPILE_FLAGS "-pedantic ${GCC_COVERAGE_COMPILE_FLAGS}")
endif()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${GCC_COVERAGE_COMPILE_FLAGS}" )
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GCC_COVERAGE_COMPILE_FLAGS}" )

if(AWNN_USE_OPENMP)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
		message("Use OPenMP")
endif()

project(awnn LANGUAGES C CXX)

## cuda
if(USE_CUDA)
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -O3 -DCONFIG_DEBUG -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70)
  set(CUDA_HOST_COMPILATION_CPP ON)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_VERBOSE ON)
  set(CUDA_64_BIT_DEVICE_CODE ON CACHE STRING "Compile device code in 64 bit mode" FORCE)

  message(STATUS "FoundCUDA              : ${CUDA_FOUND}")
  message(STATUS "Cuda cublas libraries  : ${CUDA_CUBLAS_LIBRARIES}")

  set(CMAKE_CUDA_COMPILER "nvcc")

endif(USE_CUDA)

include(CheckIncludeFiles)

#TODO support MLK? source /opt/intel/mkl/bin/mklvars.sh intel64")
if(USE_OPENBLAS)
  CHECK_INCLUDE_FILES("openblas/cblas.h" HAVE_CBLAS_OPENBLAS)  # ubuntu 1604
  CHECK_INCLUDE_FILES("cblas.h" HAVE_CBLAS) # my arch
  if(HAVE_CBLAS_OPENBLAS)
    message("has cblas in ubuntu!")
    set(AWNN_DEP_LIBS ${AWNN_DEP_LIBS}  blas)
  elseif(HAVE_CBLAS)
    message("has cblas in archlinux!")
    set(AWNN_DEP_LIBS ${AWNN_DEP_LIBS} cblas blas)

  else()
    message(FATAL_ERROR "in sievert, run:
    [ubuntu] sudo apt-get install libopenblas-dev
    [arch]: sudo pacman cblas openblas")
  endif()
endif()

configure_file (
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)

include_directories(include ${PROJECT_BINARY_DIR})

if(USE_NNPACK)
  set(NNPACK_ROOT extern/NNPACK)

  include_directories(deps/pthreadpool/include ${NNPACK_ROOT}/include/)

  # I didn't added this in extern since RELEASE flag was disabled there
  add_executable(nnpack-bench-conv
    ${NNPACK_ROOT}/bench/convolution.c
    ${NNPACK_ROOT}/bench/memread.c
    ${NNPACK_ROOT}/bench/median.c
    ${NNPACK_ROOT}/bench/perf_counter.c)
  target_include_directories(nnpack-bench-conv PRIVATE ${NNPACK_ROOT}/bench)
  target_link_libraries(nnpack-bench-conv nnpack)
  set(AWNN_DEP_LIBS ${AWNN_DEP_LIBS} nnpack nnpack_reference_layers)

endif()
list(APPEND awnn_core_SOURCES
  src/tensor.c
  src/tensor_op.c
	src/im2col.c
  src/layer_conv.c
  src/layer_conv_per_img.c
  src/layer_pool.c
  src/layer_fc.c
  src/layer_relu.c
  src/loss_softmax.c
  src/net.c
  src/net_mlp.c
  src/layer_sandwich.c
  src/solver.c
  src/net_resnet.c
  )

if (USE_NNPACK)
  list(APPEND awnn_core_SOURCES
    src/layer_conv_nnpack.c)
endif ()

add_library(awnn_core ${awnn_core_SOURCES})

target_include_directories(awnn_core PUBLIC include)
target_link_libraries(awnn_core PUBLIC ${AWNN_DEP_LIBS})

# This is a cpp lib
add_library(awnn_utils
  utils/data_cifar.cpp
  utils/weight_init.cpp
  utils/debug.cpp)

target_link_libraries(awnn_utils awnn_core)

set(AWNN_LIBS "awnn_core" "awnn_utils")

if(USE_CUDA)
  cuda_add_library(awnndevicelib
    src/tensor.cu
    src/layer_pool.cu)

  set(AWNN_DEVICE_LIBS "awnndevicelib")
endif(USE_CUDA)

if(USE_CUDA AND USE_CUDNN)
  add_subdirectory(examples/mnistCUDNN)
endif()

# nnpack and gtest
add_subdirectory(extern)

if(BUILD_TEST)
  enable_testing()
  add_subdirectory(tests)
endif()
