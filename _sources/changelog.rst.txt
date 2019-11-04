.. _changelog:

=========
Changelog
=========

.. note::

  Code is maintained in https://github.com/fengggli/gpu-computing-materials/

  Ask Feng for access.

Current
=======

:Date: 2019-11-04

Added
-------

* Reviews of works on hybrid parallelism(https://fengggli.github.io/ResearchDocs/topics/hybridparal/index.html#hybrid-parallelism)

  - One weird trick: data parallelism for convolution layer, model parallelism for dense layer, transformation in between. (because conv/dense layer have different computation/communication requirement.)
  - How to decide process layerout for a given batch size and a network architecture.

* Amazon neocpu(https://fengggli.github.io/ResearchDocs/journal/Fall19/Week9.html#neocpu)

  - end-to-end optimization for cpu-based inference.

* pipedream is part of microsoft fiddle project: https://www.microsoft.com/en-us/research/project/fiddle/, fiddle is targeting serveral problems:

  - How to train efficiently in a single gpu
  - How to train with multiple gpu
  - How to train with multi-tenant Clusters

* Different types of optimizations(coarse-grained, fine-grained, layer-wise, end-to-end) are discussed here (https://fengggli.github.io/ResearchDocs/journal/Fall19/Week10.html#coarse-grain-fine-grain-and-layer-wise)


Working on
-----------

1. Original communication optimal algorithm only considers forward pass of a direct convolution operation. It gives guidelines how to decide on best blocking size/ loop orders in a direction convolution, for different input settings(e.g. input image size, filter sizes, stride size, etc).
2. How to extend it to other components in a neural-net?
3. How to decide the balance of data/model parallelism.


TODO List
----------

* Use fine-grained lock to reduce contention.
* Theoretical model.

=========
Previous
=========

0.4.13
========

:Date: 2019-10-22

Added
-------
1. Explained why AWNN is slower than Intel-Caffe in Stampede2 SKX node Gibson (also SKX with avx512)

   - performance analysis results using intel vtune see (https://github.com/fengggli/gpu-computing-materials/issues/57)
   - AWNN still has worse single-threaded performance, most of the elapsed time is spent on im2col and col2im, since they are not currently vectorized.
   - intel-caffe uses mkldnn  JIT avx code generation to accelerate operations like convolution/pooling.
   - A SC18 paper describes some of the optimizations used in MKL-DNN(e.g. vectorization, cache/register blocking, loop reordering, kernel streaming, software prefetching, layer fusion, etc:  https://dl.acm.org/citation.cfm?id=3291744)

2. Followed several suggestions from intel performance guide, improved single-thread forward/backward time from 540 to 380ms(https://github.com/fengggli/gpu-computing-materials/issues/57#issuecomment-540705655).
3. We can add those optimization implemented in MKLDNN, (e.g. vectorization of im2col/col2im). But such optimizations are not urgent.
4. Some literature on pipeline parallelism (https://fengggli.github.io/ResearchDocs/topics/pipeline/pipeline.html#pipeline), it's a form of model parallelism.

0.4.12
========

:Date: 2019-10-07

Added
------

* Performance comparision with Intel-caffe in skx and knl nodes and corresponding analysis.

  - Performance of intel-caffe is x3.9 faster than awnn in stampede skx(https://github.com/fengggli/gpu-computing-materials/issues/54#issuecomment-537741399), not consistent with the sievert results.
  - Now I am able to build caffe using preloaded dependencies in stampede2. Need to profile to understand the inconsistent performance in stampedede2.
  - Also need to do same set of experiments in gibson.


0.4.11
=======

:Data 2019-09-26

Added
--------

1. Add worker threads support, details are at: https://github.com/fengggli/gpu-computing-materials/issues/54
2. reorganize code-structure, so that:

   * each type of layer is now associated with a "layer_setup" function, which can infer the size of output tensor and working memory based on the layer below it.
   * all working memory and middle-layer output memory are preallocated during the "set_up" phase, instead allocated/free during forward/backward
   * improved implementations of layers like fc/relu/pool to reduce extra memory copies.
   * x1.77 speedup, using float32(in sievert).


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
