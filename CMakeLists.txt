cmake_minimum_required(VERSION 3.8)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(awnn LANGUAGES C)


option(USE_CLANG OFF)
if(USE_CLANG)
  set(CMAKE_C_COMPILER "clang")
  set(CMAKE_CXX_COMPILER "clang++")
else()
  set(CMAKE_C_COMPILER "gcc")
  set(CMAKE_CXX_COMPILER "g++")
endif()


add_library(awnnlib STATIC
  src/tensor.c
  src/layer_conv.c

  include/awnn/common.h
  include/awnn/layer_conv.h
  include/awnn/tensor.h
)

target_include_directories(awnnlib PUBLIC include)
