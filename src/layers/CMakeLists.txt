add_library(awnn_layers
  layer_common.cpp
  resnet.cpp

  resnet_hybrid.cpp
  vggnet.cpp
  parallel.cpp
  )

target_link_libraries(awnn_layers pthreadpool)
