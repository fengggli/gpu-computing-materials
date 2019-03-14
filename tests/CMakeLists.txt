
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


