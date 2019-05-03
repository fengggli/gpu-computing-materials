4th discussion
===============

Notes for Feng
--------------

1. change dot matmul. like it is numpy.
2. change ifdef to pragma once
3. use const for input parameters like in layer_conv (done)

Tasks
------

GPU memory copy for tensor, you can a look the blob implementation.

Feng Li will add tensor_t in gpu.
  1. make_copy_gpu.

Yuankun and Chris will implement: (Finish around April 1)
  1. The pooling layer. (understand the code structure)
  2. The convolution layer, takes the tensor_t from gpu. ()
