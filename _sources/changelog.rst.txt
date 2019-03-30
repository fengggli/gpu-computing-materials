.. _changelog:

========
Current
========

TODO List
----------

* Layers

  * conv2d
  * batchnorm

* Net

  1. residual blocks
  2. siplified resnet

* Utility

  1. sgd
  2. reporting

* Initializer (kaiming initialization)

* Data

* Others

Working in progress:
---------------------

1. Conv2d and global pool in gpu (Chris and Yuankun)
2. 2-layer mlp (Feng).

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

=========
Previous
=========

< 0.4.1
========

see dl-docs for changelog prior to 0.4.1
