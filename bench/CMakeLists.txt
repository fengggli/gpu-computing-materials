add_executable(bench-net-resnet bench_net_resnet.cpp)
target_link_libraries(bench-net-resnet ${AWNN_LIBS} gtest)
add_test(bench-net-resnet bench-net-resnet)

