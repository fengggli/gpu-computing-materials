.. _awnn_layer-conv:

Convolution Layer
===================

1. A introduction is in here (http://cs231n.github.io/convolutional-networks/#conv), 
   The animation  is helpful for you to understand how convolution works in multi-channels.
2. The interface in numpy looks like this(https://github.com/fengggli/cs231n-assignments/blob/d4cbe582a794a5b33d81a1ecdb64f1fd3844eaaa/assignment2/cs231n/layers.py#L450)
3. A common implementation of the forwarding in the above interface: 
   see the conv_forward_im2col in `numpy fast implementation of layers <https://github.com/fengggli/cs231n-assignments/blob/d4cbe582a794a5b33d81a1ecdb64f1fd3844eaaa/assignment2/cs231n/fast_layers.py#L14>`_.
