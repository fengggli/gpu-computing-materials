# skeleton example of google-test

add_executable(test-skeleton test_skeleton.cpp)
target_link_libraries(test-skeleton ${AWNN_LIBS} gtest)
add_test(test-skeleton test-skeleton)

add_executable(test-dim test_dim.cpp)
target_link_libraries(test-dim ${AWNN_LIBS} gtest)
add_test(test-dim test-dim)

add_executable(test-tensor test_tensor.cpp)
target_link_libraries(test-tensor ${AWNN_LIBS} gtest)
add_test(test-tensor test-tensor)

add_executable(test-tensor-op test_tensor_op.cpp)
target_link_libraries(test-tensor-op ${AWNN_LIBS}  gtest)
add_test(test-tensor-op test-tensor-op)

add_executable(test-layer-conv test_layer_conv.cpp)
target_link_libraries(test-layer-conv ${AWNN_LIBS} gtest)
add_test(test-layer-conv test-layer-conv)

if(USE_CUDA)
    add_executable(test-layer-conv-device test_layer_conv_device.cpp)
    target_link_libraries(test-layer-conv-device ${AWNN_LIBS} ${AWNN_DEVICE_LIBS} gtest)
    add_test(test-layer-conv-device test-layer-conv-device)

    add_executable(test-resblock-device test_resblock_device.cpp)
    target_link_libraries(test-resblock-device ${AWNN_LIBS} ${AWNN_DEVICE_LIBS} gtest)
    add_test(test-resblock-device test-resblock-device)


    add_executable(test-cublas-device-op test_cublas_device_op.cpp)
    target_link_libraries(test-cublas-device-op ${AWNN_LIBS} ${AWNN_DEVICE_LIBS} cudart cublas gtest)
    add_test(test-cublas-device-op test-cublas-device-op)

    add_executable(test-device-util test_device_utils.cpp)
    target_link_libraries(test-device-util ${AWNN_LIBS} ${AWNN_DEVICE_LIBS} cudart cublas gtest)
    add_test(test-device-util test-device-util)

endif(USE_CUDA)

if(USE_CUDA AND USE_CUDNN)
add_executable(test-layer-conv-cudnn test_layer_conv_cudnn.cpp)
target_link_libraries(test-layer-conv-cudnn ${AWNN_LIBS}  ${AWNN_DEVICE_LIBS} gtest)
add_test(test-layer-conv-cudnn test-layer-conv-cudnn)


endif()

if (USE_NNPACK)
add_executable(test-layer-conv-nnpack test_layer_conv_nnpack.cpp)
target_link_libraries(test-layer-conv-nnpack ${AWNN_LIBS} gtest)
add_test(test-layer-conv-nnpack test-layer-conv-nnpack)
endif ()

add_executable(test-layer-conv-perimg test_layer_conv_perimg.cpp)
target_link_libraries(test-layer-conv-perimg ${AWNN_LIBS} gtest)
add_test(test-layer-conv-perimg test-layer-conv-perimg)

add_executable(test-layer-pool test_layer_pool.cpp)
target_link_libraries(test-layer-pool ${AWNN_LIBS} ${AWNN_DEVICE_LIBS} gtest)
add_test(test-layer-pool test-layer-pool)

add_executable(test-layer-fc test_layer_fc.cpp)
target_link_libraries(test-layer-fc ${AWNN_LIBS}  gtest)
add_test(test-layer-fc test-layer-fc)

add_executable(test-utility test_utility.cpp)
target_link_libraries(test-utility ${AWNN_LIBS}  gtest)
add_test(test-utility test-utility)

add_executable(test-layer-relu test_layer_relu.cpp)
target_link_libraries(test-layer-relu ${AWNN_LIBS} ${AWNN_DEVICE_LIBS}  gtest)
add_test(test-layer-relu test-layer-relu)

add_executable(test-loss-softmax test_loss_softmax.cpp)
target_link_libraries(test-loss-softmax ${AWNN_LIBS}  gtest)
add_test(test-loss-softmax test-loss-softmax)

add_executable(test-net-mlp test_net_mlp.cpp)
target_link_libraries(test-net-mlp ${AWNN_LIBS}  gtest)
add_test(test-net-mlp test-net-mlp)

add_executable(test-net-mlp-cifar test_net_mlp_cifar.cpp)
target_link_libraries(test-net-mlp-cifar ${AWNN_LIBS}  gtest)
add_test(test-net-mlp-cifar test-net-mlp-cifar)

add_executable(test-dtype-list test_dtype_list.cpp)
target_link_libraries(test-dtype-list ${AWNN_LIBS}  gtest)
add_test(test-dtype-list test-dtype-list)

add_executable(test-data-cifar test_data_cifar.cpp)
target_link_libraries(test-data-cifar ${AWNN_LIBS}  gtest)
add_test(test-data-cifar test-data-cifar)

add_executable(test-layer-sandwich test_layer_sandwich.cpp)
target_link_libraries(test-layer-sandwich ${AWNN_LIBS}  gtest)
add_test(test-layer-sandwich test-layer-sandwich)

add_executable(test-net-resnet test_net_resnet.cpp)
target_link_libraries(test-net-resnet ${AWNN_LIBS}  gtest)
if(NOT IS_CI_BUILD)
  add_test(test-net-resnet test-net-resnet)
endif()

#set_target_properties(test-net-resnet
  #PROPERTIES COMPILE_DEFINITIONS "CONFIG_DEBUG")

add_executable(test-net-resnet-cifar test_net_resnet_cifar.cpp)
target_link_libraries(test-net-resnet-cifar ${AWNN_LIBS} gtest)
# add_test(test-net-resnet-cifar test-net-resnet-cifar) # this uses a lot of mem

# test memory leaks
add_executable(test-net-resnet-mem test_net_resnet_mem.cpp)
target_link_libraries(test-net-resnet-mem ${AWNN_LIBS} gtest)

#add_executable(test-loss-softmax-simple test_loss_softmax_simple.cpp)
#target_link_libraries(test-loss-softmax-simple ${AWNN_LIBS}  gtest)
#add_test(test-loss-softmax-simple test-loss-softmax-simple)
