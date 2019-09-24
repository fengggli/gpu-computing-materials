.. _changelog:

=========
Changelog
=========

.. note::

  Code is maintained in https://github.com/fengggli/gpu-computing-materials/

  Ask Feng for access.


Current
=======

:Data 2019-08-23

Added
--------

1. Add worker threads support, details are at: https://github.com/fengggli/gpu-computing-materials/issues/54

Working on
------------

1. Reorganize memory allocation code.
2. Use fine-grained lock to reduce contention.

TODO List
----------

* Compare with intel-caffe.
* Theoretical model.

=========
Previous
=========

0.4.10
========

:Data 2019-08-23

Added
--------

1. model, extended resnet with 3 stages: 
   * previous simple model: http://ethereon.github.io/netscope/#/gist/64b013d6fee840473edc1a9a444e22ca
   * new 14-layer model: http://ethereon.github.io/netscope/#/gist/b14a68b31b3973c68b38dfc2f73d2d10


0.4.9
======
:Data 2019-06-27

Added
--------
1. Adding downsampling in the beginning of stage 3,4,5, more details see https://github.com/fengggli/gpu-computing-materials/issues/51, ignoring the boundries.
2. Residual blocks using with downsampling support and its tests.
3. Add resnet14, made of 3 stages, each stage containing 2 residual blocks.



0.4.8
======
:Data 2019-05-12

* Add nnpack support, resnet can use nnpack backend for the convolution operations(https://github.com/fengggli/gpu-computing-materials/pull/41)
* Initial implementation of convolution is slow due to explict transpose and memory copies. (https://github.com/fengggli/gpu-computing-materials/pull/41#issuecomment-486513801), we did performance analysis and improvement for the convolution layer.
* Add per-image convolution like in Caffe(https://github.com/fengggli/gpu-computing-materials/pull/49).
* There is also a comparision of AWNN vs caffe in the case of (1)NNPACK or (2)per-img im2col+openblas gemm when different batch sizes are used (https://github.com/fengggli/gpu-computing-materials/pull/49#issuecomment-490657411): Our implementation is slightly faster than Caffe when using openblas gemm; nnpack in caffe patch doesn't provide backward implementation, I can add it though.

0.4.7
======
:Data 2019-04-22

* Simplified resnet(https://github.com/fengggli/gpu-computing-materials/pull/38)
* Fix memory leaks, and some obvious optimization.
* Initializer (kaiming initialization)

0.4.6
======
:Data 2019-04-15

Added
-------

* residual block and simple resnet. See https://github.com/fengggli/gpu-computing-materials/pull/37.

0.4.5
======

:Date 2019-04-10

Added
-------

* utils for debug use (tensor mean/std, etc)
* fixed several bugs
* utils to report statistics during training(loss, train/val accuracy).
* results of mlp is in https://github.com/fengggli/gpu-computing-materials/pull/27/


0.4.4
======

:Date 2019-04-08

Added
-------

1. cifar Data loader:

  * Use data/cifar10/get_cifar10.sh to download data.
  * preprocess: normailzed, and with channel mean substracted.
  * train/validation split

2. Solver(main for loop):

  * feed batches from loader, forward/backward and gradient updates(test/test_net_mlp_cifar)

2. Weight init

  * Kaiming init and weight-scale based init.
  * Extract this part to utils/ since we use distribution from stl.

3. Doc

  * Added the network memory allocation figure.

4. Cuda

  * naiive CUDA pooling layer, set USE_CUDA=on to enable

0.4.3
=======

:Date 2019-04-01

See (https://github.com/fengggli/gpu-computing-materials/pull/19)

Added
-----------

* a fc_relu sandwich layer
* weight initialization (currently only linspace is used)
*  macro: tensor_for_each_entry in tensor.h
* net-mlp:

  - inference-only forward - mlp_forward
  - loss function to update the gradients mlp_loss
  - forward compared with numpy version
  - backward checked with numerical results
  - regulizer is  added

Changed
--------

* changed the layer cache, now each layer has a lcache_t, which can be assessed as a stack using lcache_push, and lcache_pop. See docs/source/memory.rst for more details

others
------

* clangformat using google style


0.4.2
======

:Date 2019-03-30

Added
-------

1. Layers:

  * fully-connected
  * global avg pool.
  * relu
  * softmax

2. Data structure

  * The param_t uses linux-kernel style linked list, which can be also used to construct other basic data structures like stack/queue.
  * currently it's used to manage all learnable params of fc layers.



< 0.4.1
========

see dl-docs for changelog prior to 0.4.1
