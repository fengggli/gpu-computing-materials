add_executable(test-layers test_layers.cpp)
target_link_libraries(test-layers awnn_layers ${AWNN_COMMON} ${AWNN_LIBS} gtest)

add_executable(test-topology test_topology.cpp)
target_link_libraries(test-topology awnn_layers ${AWNN_COMMON} ${AWNN_LIBS} gtest)
