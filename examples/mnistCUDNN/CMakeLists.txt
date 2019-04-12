# add file search path

set(PATH_CUDNN_MNIST  "/usr/src/cudnn_samples_v7/mnistCUDNN" CACHE PATH "path to cudnn example, avaliable in sievert")

include_directories(${PATH_CUDNN_MNIST}/FreeImage/include)
find_library(LIB_FREEIMAGE freeimage ${PATH_CUDNN_MNIST}/FreeImage/lib/linux/x86_64/)
list(APPEND mnistCUDNN_SOURCES
    ${PATH_CUDNN_MNIST}/error_util.h
    ${PATH_CUDNN_MNIST}/fp16_dev.cu
    ${PATH_CUDNN_MNIST}/fp16_dev.h
    ${PATH_CUDNN_MNIST}/fp16_emu.h
    ${PATH_CUDNN_MNIST}/fp16_emu.cpp
    ${PATH_CUDNN_MNIST}/gemv.h
    ${PATH_CUDNN_MNIST}/mnistCUDNN.cpp
)

cuda_add_executable(mnistCUDNN ${mnistCUDNN_SOURCES})
target_link_libraries(mnistCUDNN cudart cudnn cublas ${LIB_FREEIMAGE})
