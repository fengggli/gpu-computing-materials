add_library(awnn_layers
  layer_common.cpp
  resnet.cpp
  parallel.cpp
  )
target_link_libraries(awnn_layers pthreadpool)
