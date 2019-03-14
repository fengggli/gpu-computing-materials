
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

