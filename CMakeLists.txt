cmake_minimum_required(VERSION 3.5)
#set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_CLANG "Use clang compiler set" OFF)
option(USE_OPENBLAS "USE openblas for blas" OFF)
option(BUILD_TEST "Build google test and all test cases" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

if(USE_CLANG)
  set(CMAKE_C_COMPILER "clang")
  set(CMAKE_CXX_COMPILER "clang++")
else()
  set(CMAKE_C_COMPILER "gcc")
  set(CMAKE_CXX_COMPILER "g++")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_SHARED_LINKER_FLAGS "-rdynamic") 

project(awnn LANGUAGES C CXX)

include(CheckIncludeFiles)

#TODO support MLK? source /opt/intel/mkl/bin/mklvars.sh intel64")
if(USE_OPENBLAS)
  CHECK_INCLUDE_FILES("cblas.h" HAVE_CBLAS)
  if(HAVE_CBLAS)
    message("has cblas!")
  else()
    message(FATAL_ERROR "in sievert, run:
    sudo apt-get install libopenblas-dev")
  endif()
  set(AWNN_LIBS ${AWNN_LIBS} openblas)
endif()

configure_file (
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)

include_directories(include ${PROJECT_BINARY_DIR})

add_library(awnnlib SHARED
  src/tensor.c
  src/tensor_op.c
  src/layer_conv.c
  src/layer_pool.c
  src/layer_fc.c
  src/layer_relu.c
  src/loss_softmax.c
  src/net_mlp.c)

target_link_libraries(awnnlib PUBLIC ${AWNN_LIBS})

if(BUILD_TEST)
#  add_subdirectory(/usr/src/gtest ${CMAKE_CURRENT_BINARY_DIR}/gtest)
  add_subdirectory(extern/googletest extern/googletest)
  enable_testing()
  add_subdirectory(tests)
endif()

target_include_directories(awnnlib PUBLIC include)
