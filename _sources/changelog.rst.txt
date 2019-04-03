.. _changelog:

=========
Changelog
=========

Current
=======

.. note::

  Code is maintained in https://github.com/fengggli/gpu-computing-materials/

  Ask Feng for access.

TODO List
----------

* Utility:

  show show backtrace (https://www.gnu.org/software/libc/manual/html_node/Backtraces.html)

* Layers

  * conv2d
  * batchnorm

* Net

  1. Lenet
  2. residual blocks
  3. simplified resnet

* Utility

  1. solver
  2. reporting

* Initializer (kaiming initialization)

* Data

* Others


Working in progress
--------------------

1. Conv2d and global pool in gpu (Chris and Yuankun).
2. solver.

Added
======

* naiive CUDA pooling layer, set USE_CUDA=on to enable

=========
Previous
=========

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

  - inference-only forward - mlp_scores
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
