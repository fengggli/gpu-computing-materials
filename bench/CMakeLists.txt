add_executable(bench-net-resnet bench_net_resnet.cpp)
target_link_libraries(bench-net-resnet ${AWNN_LIBS} gtest)
add_test(bench-net-resnet bench-net-resnet)

if(USE_MKL)
add_executable(bench-conv bench_conv.cpp)
target_compile_options(bench-conv PRIVATE "-qopenmp")
target_link_libraries(bench-conv ${AWNN_LIBS} gtest)
add_test(bench-conv bench-conv)
endif()


