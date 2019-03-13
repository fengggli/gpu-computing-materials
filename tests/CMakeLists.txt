
# skeleton example of google-test
add_executable(test-skeleton test_skeleton.cpp)
target_link_libraries(test-skeleton awnnlib gtest)

add_executable(test-dim test_dim.cpp)
target_link_libraries(test-dim awnnlib gtest)

