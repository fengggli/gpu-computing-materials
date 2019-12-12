cmake_minimum_required(VERSION 3.5)
#set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(IS_CI_BUILD "Is a build for CI" OFF) # Reduce problem size if it is run in CI
option(USE_CLANG "Use clang compiler set" OFF)

# blas
option(USE_MKL "Use intel compiler set" OFF)
option(USE_OPENBLAS "USE openblas for blas" OFF)
option(USE_SEQUENTIAL_BLAS "USE openblas for blas" ON)

option(BUILD_TEST "Build google test and all test cases" ON)
option(USE_CUDA "Build with CUDA" OFF)
option(DEBUG_GPU_FUNC_ENTRY_NOTIFY "Print device function names upon function call" ON)
option(USE_CUDNN "Build with CUDNN" OFF)
option(USE_NNPACK "Build with NNPACK" OFF)
option(AWNN_USE_FLT32 "Use single precision floating point" OFF)
option(GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC "Count allocations and destroy globally" OFF)
option(AWNN_USE_OPENMP "Use OpenMP" OFF)
option(USE_MEMCHECK "Use mem santilizer to check leaks" OFF)
option(USE_BOOST_STACKTRACE "Use booststack trace" OFF)

option(USE_ICC "Use intel compiler(icc, icpc)" OFF)
option(USE_AVX512 "use avx512, avail in stampede SKX and KNL" OFF)
#set (AWNN_GEMM_THREADS 1 CACHE STRING "number of openblas threads")

option(USE_CLANG "Use clang compiler set" OFF)
option(BUILD_TEST "Build google test and all test cases" ON)
option(USE_STRICT_CHECK "use pedantic flag" OFF)

set(MKL_ROOT "/opt/intel/mkl/" CACHE PATH "where to find mkl")

if (NOT AWNN_USE_FLT32 AND USE_NNPACK)
  message(FATAL_ERROR 'nnpack only support float32')
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# for -O3
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

if(USE_CLANG)
  set(CMAKE_C_COMPILER "clang")
  set(CMAKE_CXX_COMPILER "clang++")
elseif(USE_ICC)
  set(CMAKE_C_COMPILER "icc")
  set(CMAKE_CXX_COMPILER "icpc")
else()
  set(CMAKE_C_COMPILER "gcc")
  set(CMAKE_CXX_COMPILER "g++")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wextra -Wconversion -rdynamic")
if(USE_STRICT_CHECK)
  set(GCC_COVERAGE_COMPILE_FLAGS "-pedantic ${GCC_COVERAGE_COMPILE_FLAGS}")
endif()

if(USE_MEMCHECK)
  set(GCC_COVERAGE_COMPILE_FLAGS "-fsanitize=address -fno-omit-frame-pointer ${GCC_COVERAGE_COMPILE_FLAGS}")
endif()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${GCC_COVERAGE_COMPILE_FLAGS}" )
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GCC_COVERAGE_COMPILE_FLAGS}" )

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.13.0")
  # add_link_options("-no-pie") # to give line number in stacktrace
else()
  message("Debug Build stracktrace without line number(${CMAKE_VERSION} <= 3.13")
endif()


if(AWNN_USE_OPENMP)
  message(FATAL_ERROR "openmp support is not complete yet")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  message("Use OPenMP")
endif()

if(USE_MKL)
  if(USE_ICC)
    set(MKL_SEQUENTIAL_LINK_OPTIONS "-mkl=sequential")
    if(USE_AVX512)
      set(MKL_COMPILE_OPTIONS "-xCORE-AVX512")
    endif()

    #elseif("${MKL_ROOT}" STREQUAL "/opt/intel/mkl/")
  else()
    set(MKL_LIB_PATH "${MKL_ROOT}/lib/intel64")

    set(MKL_SEQUENTIAL_LINK_OPTIONS "-L${MKL_LIB_PATH} -Wl,-rpath,${MKL_LIB_PATH},--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")
    set(MKL_COMPILE_OPTIONS "-m64  -I${MKL_ROOT}/include")
    if(USE_AVX512)
      set(MKL_COMPILE_OPTIONS "-march=skylake-avx512 ${MKL_COMPILE_OPTIONS}")
    else()
      #set(MKL_COMPILE_OPTIONS "-march=native -fopt-info-vec ${MKL_COMPILE_OPTIONS}")
      set(MKL_COMPILE_OPTIONS "-march=native ${MKL_COMPILE_OPTIONS}")

    endif()

    set(MKL_IOMP_LINK_OPTIONS " -L${MKL_LIB_PATH} -Wl,-rpath,${MKL_LIB_PATH},--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")

#  else() # use caffe's builtin mkl
    #message("MKL_ROOT is ${MKL_ROOT}")
    #set(MKL_LIB_PATH "${MKL_ROOT}/lib/")
    ## sequential version not bounded
    #set(MKL_SEQUENTIAL_LINK_OPTIONS " -L${MKL_LIB_PATH} -Wl,-rpath,${MKL_LIB_PATH},--no-as-needed -lmklml_intel -liomp5 -lm -ldl")
    ##set(MKL_SEQUENTIAL_LINK_OPTIONS "-L${MKL_LIB_PATH} -Wl,-rpath,${MKL_LIB_PATH},--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")
    #set(MKL_COMPILE_OPTIONS "-m64 -I${MKL_ROOT}/include")
    #set(MKL_IOMP_LINK_OPTIONS " -L${MKL_LIB_PATH} -Wl,-rpath,${MKL_LIB_PATH},--no-as-needed -lmklml_intel -liomp5 -lm -ldl")
   endif()
endif()


set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MKL_COMPILE_OPTIONS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MKL_COMPILE_OPTIONS}")

if(USE_SEQUENTIAL_BLAS)
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MKL_SEQUENTIAL_LINK_OPTIONS}")
	message("Use non-threaded MKL")
else()
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MKL_IOMP_LINK_OPTIONS}")
	message("Use multi-threads(Intel OpenMP) MKL")
endif()


project(awnn LANGUAGES C CXX)

## cuda
if(USE_CUDA)
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -O3 -DCONFIG_DEBUG -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70)
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

include_directories(include ${PROJECT_BINARY_DIR} src/)

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
  src/memory.c
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

add_subdirectory(extern/pthreadpool)
include_directories(extern/pthreadpool/include)

set(AWNN_LIBS "awnn_core" "awnn_utils" dl)

if(USE_CUDA)
  list(APPEND awnndevicelib_SOURCES
      include/awnndevice/range.cuh
      include/awnndevice/device_utils.cuh
      include/awnndevice/layer_conv_device.cuh
      include/awnndevice/layer_sandwich_device.cuh
      include/awnndevice/cublas_wrappers.cuh
      include/awnndevice/device_utils_harness.cuh

      src-device/tensor.cu
      src-device/layer_pool.cu
      src-device/layer_conv_device.cu
      src-device/layer_sandwich_device.cu
      src-device/cublas_wrappers.cu
      src-device/device_utils_harness.cu
  )

  if(USE_CUDNN)
    list(APPEND awnndevicelib_SOURCES
        src/layer_conv_cudnn.cu)
    add_subdirectory(examples/mnistCUDNN)
  endif(USE_CUDNN)

  cuda_add_library(awnndevicelib ${awnndevicelib_SOURCES})


  target_link_libraries(awnndevicelib cudart cublas)

  if(USE_CUDNN)
    target_link_libraries(awnndevicelib cudnn)
  endif(USE_CUDNN)

  set(AWNN_DEVICE_LIBS "awnndevicelib")
endif(USE_CUDA)

add_subdirectory(src/awnn)

# nnpack and gtest
add_subdirectory(extern)

if(BUILD_TEST)
  enable_testing()
  add_subdirectory(tests)
  if(NOT IS_CI_BUILD)
    add_subdirectory(bench)
  endif()
endif()
