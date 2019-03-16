.. _awnn_layer_pool:

Pooling Layer
==============

1. A introduction pooling layer is in here (http://cs231n.github.io/convolutional-networks/#pool)
2. In Resnet, we only need to implement a global average pool: which downsample each channel into one value([N,C,H,W] -> [N,C,1,1]). 
   Take a looked at `a reference global average pool implementation in numpy <https://github.com/fengggli/dl-docs/blob/7559c066f2a3a9740fa093271cf8dc0623a679bf/python/conn/layers/basic_layers.py#L528>`_. (it's private repo but i have sent requests to add you as collaborators.).
   Also takes a look at the backward pass.
