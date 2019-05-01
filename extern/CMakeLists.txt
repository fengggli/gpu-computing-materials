# use default flags for external library
set(CMAKE_C_FLAGS_DEBUG "" )
set(CMAKE_CXX_FLAGS_DEBUG "")

set(CMAKE_C_FLAGS "" )
set(CMAKE_CXX_FLAGS "")

if(USE_NNPACK)
  add_subdirectory(NNPACK NNPACK)
endif()

if(BUILD_TEST AND NOT TARGET gtest) ## nnpack also build gtest as deps
  add_subdirectory(googletest googletest)
endif()
