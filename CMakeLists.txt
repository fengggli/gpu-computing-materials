cmake_minimum_required(VERSION 3.5)
#set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_CLANG "Use clang compiler set" OFF)
option(BUILD_TEST "Build google test and all test cases" ON)

if(USE_CLANG)
  set(CMAKE_C_COMPILER "clang")
  set(CMAKE_CXX_COMPILER "clang++")
else()
  set(CMAKE_C_COMPILER "gcc")
  set(CMAKE_CXX_COMPILER "g++")
endif()

set(CMAKE_CXX_STANDARD 11)

project(awnn LANGUAGES C CXX)

add_library(awnnlib STATIC
  src/tensor.c
  src/layer_conv.c

  include/awnn/common.h
  include/awnn/layer_conv.h
  include/awnn/tensor.h
)

if(BUILD_TEST)
  #add_subdirectory(/usr/src/gtest ${CMAKE_CURRENT_BINARY_DIR}/gtest)
  add_subdirectory(extern/googletest extern/googletest)
  enable_testing()
  add_subdirectory(tests)
endif()

target_include_directories(awnnlib PUBLIC include)
