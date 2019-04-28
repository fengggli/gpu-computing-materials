.. _awnn_batch-norm:

Batch normalization
====================

1. I need two transpose function like Chris does for conv2d.
   This makes spatial bn (N, C, H, W)  -> (NxHxW, C)

2. I need some matrix, vector operations, where the vector has the size of (C,)
