add_executable(test-layers test_layers.cpp)
target_link_libraries(test-layers awnn_layers ${AWNN_COMMON} ${AWNN_LIBS} gtest)
