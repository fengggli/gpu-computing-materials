# skeleton example of google-test

add_executable(test-skeleton test_skeleton.cpp)
target_link_libraries(test-skeleton awnnlib gtest)
add_test(test-skeleton test-skeleton)

add_executable(test-dim test_dim.cpp)
target_link_libraries(test-dim awnnlib gtest)
add_test(test-dim test-dim)

add_executable(test-tensor test_tensor.cpp)
target_link_libraries(test-tensor awnnlib gtest)
add_test(test-tensor test-tensor)

add_executable(test-tensor-op test_tensor_op.cpp)
target_link_libraries(test-tensor-op awnnlib  gtest)
add_test(test-tensor-op test-tensor-op)

add_executable(test-layer-conv test_layer_conv.cpp)
target_link_libraries(test-layer-conv awnnlib  gtest)
add_test(test-layer-conv test-layer-conv)

add_executable(test-layer-pool test_layer_pool.cpp)
target_link_libraries(test-layer-pool awnnlib awnnlib_device gtest)
add_test(test-layer-pool test-layer-pool)

add_executable(test-layer-fc test_layer_fc.cpp)
target_link_libraries(test-layer-fc awnnlib  gtest)
add_test(test-layer-fc test-layer-fc)

add_executable(test-utility test_utility.cpp)
target_link_libraries(test-utility awnnlib  gtest)
add_test(test-utility test-utility)
set_target_properties(test-utility
  PROPERTIES COMPILE_DEFINITIONS "CONFIG_DEBUG")

add_executable(test-layer-relu test_layer_relu.cpp)
target_link_libraries(test-layer-relu awnnlib  gtest)
add_test(test-layer-relu test-layer-relu)

add_executable(test-loss-softmax test_loss_softmax.cpp)
target_link_libraries(test-loss-softmax awnnlib  gtest)
add_test(test-loss-softmax test-loss-softmax)

add_executable(test-net-mlp test_net_mlp.cpp)
target_link_libraries(test-net-mlp awnnlib  gtest)
add_test(test-net-mlp test-net-mlp)

add_executable(test-dtype-list test_dtype_list.cpp)
target_link_libraries(test-dtype-list awnnlib  gtest)
add_test(test-dtype-list test-dtype-list)

add_executable(test-solver test_solver.cpp)
target_link_libraries(test-solver awnnlib  gtest)
add_test(test-solver test-solver)


#add_executable(test-loss-softmax-simple test_loss_softmax_simple.cpp)
#target_link_libraries(test-loss-softmax-simple awnnlib  gtest)
#add_test(test-loss-softmax-simple test-loss-softmax-simple)
